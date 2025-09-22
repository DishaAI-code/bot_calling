
import os
import asyncio
import time
import uuid
from dotenv import load_dotenv

from flask import Flask, request, Response, jsonify
from fastapi import FastAPI, WebSocket
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn

from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame, LLMRunFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from starlette.websockets import WebSocketState

from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

from langfuse import get_client

# === Always enable DEBUG logging ===
os.environ["LOGURU_LEVEL"] = "DEBUG"

# === ENV ===
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
# Langfuse config
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
if LANGFUSE_SECRET_KEY:
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
langfuse = get_client() if LANGFUSE_SECRET_KEY else None

# === FLASK app for HTTP routes ===
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def health_check():
    return Response("Flask OK", content_type="text/plain")

@flask_app.route("/makecall", methods=["POST"])
def make_call():
    data = request.get_json(silent=True) or {}
    to_number = data.get("phone") or data.get("to")
    if not to_number:
        return jsonify({"success": False, "message": "Phone number required"}), 400

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{PUBLIC_BASE_URL}/flask/answer",
            machine_detection="Enable",
            status_callback=f"{PUBLIC_BASE_URL}/flask/status",
            status_callback_event=["completed"]
        )
        return jsonify({"success": True, "sid": call.sid}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@flask_app.route("/answer", methods=["POST"])
def answer_call():
    resp = VoiceResponse()
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://") + "/twilio/ws"
    resp.connect().stream(url=ws_url)
    return Response(str(resp), mimetype="text/xml")

@flask_app.route("/status", methods=["POST"])
def call_status():
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")
    logger.info(f"Call {call_sid} ended with status {call_status}")
    return ("", 204)

# === FASTAPI app ===
asgi_app = FastAPI()
asgi_app.mount("/flask", WSGIMiddleware(flask_app))

def _now():
    return time.strftime("%H:%M:%S")

# === WS for Twilio Media Stream ===
@asgi_app.websocket("/twilio/ws")
async def twilio_media_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        transport_type, call_data = await parse_telephony_websocket(websocket)
    except Exception as e:
        logger.error(f"Parse error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        return

    logger.info(f"Transport: {transport_type}, call_data: {call_data}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # services
    stt = AzureSTTService(
        api_key=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
        auto_detect_source_language=True,
        languages=["en-IN", "hi-IN", "gu-IN"]
    )
    
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    tts = ElevenLabsTTSService(api_key=ELEVEN_API_KEY, voice_id=ELEVEN_VOICE_ID)

    messages = [
        {"role": "system", "content": "You are a friendly AI assistant. Detect the user’s language and reply in that language (English, Hindi, Gujarati). Keep it short."}
    ]
    ctx = OpenAILLMContext(messages)
    ctx_agg = llm.create_context_aggregator(ctx)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        ctx_agg.user(),
        llm,
        tts,
        transport.output(),
        ctx_agg.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
        observers=[RTVIObserver(rtvi)],
    )

    # === Debug Handlers ===
    @stt.event_handler(TextFrame)
    async def on_stt(frame: TextFrame):
        ts = _now()
        text = getattr(frame, "text", None)
        lang = getattr(frame, "language", None)
        conf = getattr(frame, "confidence", None)
        print(f"[{ts}] [STT] text={text} | lang={lang} | conf={conf}", flush=True)

    @llm.event_handler(LLMRunFrame)
    async def on_llm_run(frame: LLMRunFrame):
        ts = _now()
        print(f"[{ts}] [LLM START] Prompt sent to model...", flush=True)

    @llm.event_handler(LLMFullResponseEndFrame)
    async def on_llm_response_end(frame: LLMFullResponseEndFrame):
        ts = _now()
        final_text = getattr(frame, "final_text", None)
        print("\n" + "="*50, flush=True)
        print(f"[{ts}] [LLM FINAL] {final_text}", flush=True)
        print("="*50 + "\n", flush=True)

    # Transport events
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_inst, client):
        logger.info("Twilio client connected")
        messages.append({"role": "system", "content": "Say hello to the caller."})
        await task.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_inst, client):
        logger.info("Twilio client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

# Entrypoint
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Use Azure’s port or 8000 locally
    uvicorn.run("app:asgi_app", host="0.0.0.0", port=port)

