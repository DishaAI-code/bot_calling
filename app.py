
import os
import asyncio
import requests
import time
import uuid
from dotenv import load_dotenv

# Flask (for your existing HTTP routes)
from flask import Flask, request, Response, send_from_directory, jsonify

# FastAPI (for WebSocket / Pipecat transport)
from fastapi import FastAPI, WebSocket
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn

# Twilio helpers
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

# Pipecat imports (used in the streaming pipeline)
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
# from pipecat.pipeline.observer import PipelineObserver

# --- Azure + ElevenLabs + OpenAI ---
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService



# === ENV ===
load_dotenv()

TWILIO_ACCOUNT_SID =os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_REGION")
ELEVEN_API_KEY =os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")  # Default voice

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_UR")  # Your public URL here, e.g., from ngrok



# === FLASK app for HTTP routes ===
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def health_check():
    return Response("Flask OK", content_type="text/plain")

# Outbound call
@flask_app.route("/makecall", methods=["POST"])
def make_call():
    data = request.get_json(silent=True) or {}
    to_number = data.get("phone") or data.get("to")
    if not to_number:
        return jsonify({"success": False, "message": "Phone number is required"}), 400

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

# Answer call
@flask_app.route("/answer", methods=["POST"])
def answer_call():
    call_sid = request.form.get("CallSid", "unknown")

    resp = VoiceResponse()
    # Start Twilio Media Streams → FastAPI WebSocket
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://") + "/twilio/ws"
    resp.connect().stream(url=ws_url)

    return Response(str(resp), mimetype="text/xml")

# Status callback
@flask_app.route("/status", methods=["POST"])
def call_status():
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")
    logger.info(f"Call {call_sid} ended with status {call_status}")
    return ("", 204)

# === FASTAPI app ===
asgi_app = FastAPI()
# Mount Flask under /flask → all HTTP routes live here
asgi_app.mount("/flask", WSGIMiddleware(flask_app))

# Pipecat WebSocket endpoint
@asgi_app.websocket("/twilio/ws")
async def twilio_media_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        transport_type, call_data = await parse_telephony_websocket(websocket)
    except Exception as e:
        logger.error(f"Failed to parse Twilio websocket: {e}")
        await websocket.close()
        return

    logger.info(f"Pipecat transport: {transport_type}, call_data: {call_data}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    transport_params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        vad_analyzer=SileroVADAnalyzer(),
        serializer=serializer,
    )

    transport = FastAPIWebsocketTransport(websocket=websocket, params=transport_params)

    # === Services ===
    stt = AzureSTTService(
        api_key=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
        auto_detect_source_language=True,
        languages=["en-IN", "hi-IN", "gu-IN","kn-IN"]
    )

    llm_service = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    tts = ElevenLabsTTSService(
        api_key=ELEVEN_API_KEY,
        voice_id=ELEVEN_VOICE_ID,
        stability=0.5,
        similarity_boost=0.75
    )
    
    # === Context ===
    messages = [
    {
        "role": "system",
        "content": """You are a multilingual AI voice assistant. Follow these rules strictly:
1. The user will be greeted with a greeting message when the call connects.
2. Detect the user's spoken language automatically from their speech.
3. Respond in the **same language** as the user's last detected message.
4. If the user switches language, switch immediately to that language.
5. Supported languages: English (en), Hindi (hi), Gujarati (gu), Kannada (kn).
6. Keep responses short, friendly, and conversational.
7. Do not repeat the welcome greeting - it has already been said."""
    }
]
    context = OpenAILLMContext(messages)
    context_agg = llm_service.create_context_aggregator(context)
    
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # === Pipeline ===
    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        context_agg.user(),
        llm_service,
        tts,
        transport.output(),
        context_agg.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    
    # === Debug Handlers - Register them on the individual services ===
    @stt.event_handler(TextFrame)
    async def on_stt(frame: TextFrame):
        ts = time.strftime("%H:%M:%S")
        language_detected = frame.language or "unknown"
        
        print(f"\n🎤 === STT DETECTION ===")
        print(f"[{ts}] Language: {language_detected}")
        print(f"[{ts}] Text: {frame.text}")
        print("=" * 50)

    @llm_service.event_handler(LLMRunFrame)
    async def on_llm_start(frame: LLMRunFrame):
        ts = time.strftime("%H:%M:%S")
        print(f"\n🚀 === LLM START ===")
        print(f"[{ts}] LLM processing started")
        print("=" * 50)

    @llm_service.event_handler(LLMFullResponseEndFrame)
    async def on_llm_end(frame: LLMFullResponseEndFrame):
        ts = time.strftime("%H:%M:%S")
        print(f"\n✅ === LLM COMPLETE ===")
        print(f"[{ts}] LLM response completed")
        print("=" * 50)

    @tts.event_handler(TextFrame)
    async def on_tts_input(frame: TextFrame):
        ts = time.strftime("%H:%M:%S")
        print(f"\n🔊 === TTS INPUT ===")
        print(f"[{ts}] Text to be spoken: {frame.text}")
        print("=" * 50)
    
    # @transport.event_handler("on_client_connected")
    # async def on_client_connected(transport_inst, client):
    #     logger.info("Client connected to Twilio Media Stream")
    #     # Send complete welcome message directly to TTS
    #     welcome_message = "Hello! I am your AI assistant calling from Autodesk. I can understand and speak English, Hindi, Gujarati, and Kannada. How can I help you today?"
    #     # Queue it directly as a TextFrame
    #     await task.queue_frame(TextFrame(welcome_message))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_inst, client):
        logger.info("Client connected to Twilio Media Stream")
        # Add the greeting as an assistant message to context first
        greeting = "Hello I am your AI assistant calling from Autodesk I can understand and speak English, Hindi, Gujarati, and Kannada How can I help you today?"
        # Add to context so the assistant "remembers" saying it
        messages.append({"role": "system", "content": greeting})
        # Now send it directly to TTS - try breaking it into smaller chunks
        await task.queue_frame(TextFrame(greeting))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_inst, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"WebSocket already closed: {e}")

# Entrypoint
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:asgi_app", host="0.0.0.0", port=port, reload=True)


