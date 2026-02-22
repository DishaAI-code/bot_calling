import asyncio
import base64
import logging
from dotenv import load_dotenv
import os
from time import perf_counter
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    vad,
    stt,
    utils,
    Agent,
    AgentSession,
    RoomInputOptions,   # ← ADDED
)
from livekit.plugins import elevenlabs
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import openai, silero
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, cast
import threading
import uvicorn
from livekit.agents import metrics, MetricsCollectedEvent
from livekit.agents import ConversationItemAddedEvent
from livekit.agents.llm import ImageContent, AudioContent
from livekit.agents import UserInputTranscribedEvent
from langfuse import get_client, observe
from livekit.plugins import sarvam
from rag_utils import process_pdf_and_ask, generate_general_response, query_pinecone, generate_rag_response
from livekit.agents import ChatContext, ChatMessage
from memory_service import store_user_memory, get_user_memories

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

load_dotenv(dotenv_path=".env")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.DEBUG)

stt_logger = logging.getLogger("STT")
stt_logger.setLevel(logging.DEBUG)

vad_logger = logging.getLogger("VAD")
vad_logger.setLevel(logging.DEBUG)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
sarvam_api_key = os.getenv("SARVAM_API_KEY")

if not sarvam_api_key:
    logger.error("sarvam api is not in the environment -> It is required for STT purpose")

langfuse_client = None

agent_worker_status = {
    "running": False,
    "started_at": None,
    "calls_dispatched": 0,
}

app = FastAPI(
    title="LiveKit Outbound Caller API",
    description="API endpoints for managing LiveKit outbound calls with multilingual support",
    version="3.0.0"
)


class CallRequest(BaseModel):
    phone_number: str
    room_name: Optional[str] = None


class BatchCallRequest(BaseModel):
    phone_numbers: List[str]


class CallResponse(BaseModel):
    success: bool
    message: str
    room_name: Optional[str] = None
    phone_number: str


# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return {
        "service": "LiveKit Caller API - Persistent Worker (Inbound + Outbound)",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "dispatch_call": "/dispatch/call",
            "dispatch_batch": "/dispatch/batch"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_worker": "running" if agent_worker_status["running"] else "stopped",
        "timestamp": perf_counter()
    }


@app.get("/status")
async def get_status():
    return {
        "agent_worker": agent_worker_status,
        "environment": {
            "livekit_url": os.getenv("LIVEKIT_URL", "not set"),
            "sip_trunk_configured": bool(outbound_trunk_id and outbound_trunk_id.startswith("ST_")),
        }
    }


@app.post("/dispatch/call", response_model=CallResponse)
async def dispatch_call(request: CallRequest):
    try:
        livekit_url = os.getenv("LIVEKIT_URL")
        livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not all([livekit_url, livekit_api_key, livekit_api_secret]):
            raise HTTPException(
                status_code=500,
                detail="LiveKit credentials not configured. Check environment variables."
            )

        if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
            raise HTTPException(
                status_code=500,
                detail="SIP trunk not configured. Check SIP_OUTBOUND_TRUNK_ID environment variable."
            )

        lk_api = api.LiveKitAPI(
            url=livekit_url,
            api_key=livekit_api_key,
            api_secret=livekit_api_secret,
        )

        room_name = request.room_name or f"call-{request.phone_number.replace('+', '')}-{int(perf_counter() * 1000)}"

        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        logger.info(f"Created room: {room_name}")

        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="outbound-caller",
                metadata=request.phone_number,
            )
        )

        logger.info(f"Dispatched call to {request.phone_number} in room {room_name}")
        agent_worker_status["calls_dispatched"] += 1

        await lk_api.aclose()

        return CallResponse(
            success=True,
            message=f"Call dispatched successfully to {request.phone_number}",
            room_name=room_name,
            phone_number=request.phone_number
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dispatching call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to dispatch call: {str(e)}")


@app.post("/dispatch/batch")
async def dispatch_batch_calls(request: BatchCallRequest):
    results = []
    success_count = 0

    for phone_number in request.phone_numbers:
        try:
            call_request = CallRequest(phone_number=phone_number)
            result = await dispatch_call(call_request)
            results.append({
                "phone_number": phone_number,
                "success": True,
                "room_name": result.room_name
            })
            success_count += 1
            await asyncio.sleep(2)
        except Exception as e:
            results.append({
                "phone_number": phone_number,
                "success": False,
                "error": str(e)
            })
            logger.error(f"Failed to dispatch call to {phone_number}: {e}")

    return {
        "total": len(request.phone_numbers),
        "successful": success_count,
        "failed": len(request.phone_numbers) - success_count,
        "results": results
    }


# ============ LANGFUSE TRACING SETUP ============

def setup_langfuse(host=None, public_key=None, secret_key=None):
    global langfuse_client

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not public_key or not secret_key or not host:
        logger.warning("Langfuse not configured - tracing disabled")
        return False

    try:
        langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        set_tracer_provider(trace_provider)

        if LANGFUSE_AVAILABLE:
            langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info("Langfuse tracing enabled (OpenTelemetry + SDK)")
        else:
            logger.info("Langfuse tracing enabled (OpenTelemetry only)")

        return True
    except Exception as e:
        logger.error(f"Failed to set up Langfuse tracing: {e}")
        return False


# ============ LLM INTERACTION LOGGER ============

async def log_llm_interaction(user_input: str, llm_response: str, session_metadata: dict = None):
    from langfuse import get_client

    print("=" * 80)
    print("LLM INTERACTION LOG")
    print(f"USER INPUT:\n{user_input}")
    print("-" * 80)
    print(f"AGENT RESPONSE:\n{llm_response}")
    print("=" * 80)

    try:
        langfuse_client = get_client()
        if not langfuse_client:
            return

        lf_trace_id = langfuse_client.create_trace_id()

        with langfuse_client.start_as_current_span(
            name="llm-interaction",
            trace_context={"trace_id": lf_trace_id},
            input={"user_message": user_input},
            output={"agent_response": llm_response},
            metadata=session_metadata or {},
        ) as span:
            langfuse_client.update_current_trace(tags=["llm-interaction", "voice-agent"])
            with span.start_as_current_generation(
                name="gpt4-response",
                model="gpt-4",
                input=user_input,
                output=llm_response,
                metadata={
                    "language": session_metadata.get("language", "unknown") if session_metadata else "unknown",
                },
            ):
                pass

        langfuse_client.flush()
    except Exception as e:
        logger.error(f"Langfuse error: {e}")


# ============ AGENT ============

class MultilingualAgent(Agent):
    def __init__(self, user_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_id = user_id
        logger.info("MultilingualAgent initialized with RAG support")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        try:
            user_query = new_message.text_content
            logger.info(f"[RAG] User turn completed: {user_query}")

            # 1. Inject user memory
            memories = get_user_memories(self.user_id)
            logger.info(f"[Memory] Retrieved memories: {memories}")
            if memories:
                memory_block = "\n".join(f"- {m}" for m in memories)
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "The following are important things you know about the user:\n"
                        f"{memory_block}\n\n"
                        "Use these facts when answering."
                    )
                )

            # 2. RAG context from Pinecone
            rag_response = await asyncio.to_thread(
                generate_rag_response,
                user_query,
                10
            )

            if rag_response and rag_response != "No matching chunks found in Pinecone.":
                logger.info(f"[RAG] Found relevant context: {rag_response[:200]}...")
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "The following is relevant context from the knowledge base:\n\n"
                        f"{rag_response}\n\n"
                        "Use this context to provide an accurate answer. "
                        "If the answer is not in this context, say you don't have that information."
                    )
                )
            else:
                logger.info("[RAG] No relevant context found - LLM will respond generally")

        except Exception as e:
            logger.error(f"[RAG] Error in on_user_turn_completed: {e}")
            import traceback
            traceback.print_exc()


# ============ SYSTEM PROMPTS ============

_default_instructions = """
You are a multilingual AI voice assistant.

LANGUAGE RULES:
1. Detect the language the user speaks and ALWAYS reply in the SAME language.
2. If the user switches language, switch your language too.

RAG USAGE (VERY IMPORTANT):
3. If RAG document context is provided, ALWAYS use it to answer.
4. Use document-based answers FIRST. If nothing relevant is found, say sorry, you don't know.
5. You are NOT limited to Autodesk only. You can answer ANY relevant topic.

CONVERSATION STYLE:
6. Responses must be SHORT (1–3 sentences), friendly, and conversational.
7. Do NOT repeat the welcome greeting.
8. If you do not know the answer and no document context exists, say you don't know and ask a follow-up.
"""

_greeting_message = (
    "Hello, I am your AI assistant calling from Autodesk. "
    "I can understand and speak English, Hindi, Gujarati, and Kannada. "
    "How can I help you today?"
)

_inbound_greeting_message = (
    "Hello! Thank you for calling Autodesk. I am your AI assistant. "
    "I can understand and speak English, Hindi, Gujarati, and Kannada. "
    "How can I help you today?"
)


# ============ ENTRYPOINT ============

async def entrypoint(ctx: JobContext):
    global _default_instructions, _greeting_message, _inbound_greeting_message, outbound_trunk_id

    logger.info("=" * 80)
    logger.info("AGENT ENTRYPOINT STARTED")
    logger.info("=" * 80)

    setup_langfuse()

    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    phone_number = ctx.job.metadata  # Empty = inbound, non-empty = outbound

    if phone_number:
        # ── OUTBOUND CALL ─────────────────────────────────────────────────────
        logger.info(f"OUTBOUND CALL: Dialing {phone_number} to room {ctx.room.name}")
        user_identity = "phone_user"

        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=user_identity,
            )
        )

        participant = await ctx.wait_for_participant(identity=user_identity)
        logger.info(f"Participant joined: {user_identity}")

        start_time = perf_counter()
        while perf_counter() - start_time < 30:
            call_status = participant.attributes.get("sip.callStatus")

            if call_status == "active":
                logger.info("User has picked up - call is active")
                break
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
                logger.info("User rejected the call")
                ctx.shutdown()
                return
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
                logger.info("User did not pick up")
                ctx.shutdown()
                return

            await asyncio.sleep(0.1)

        await run_voice_agent(ctx, participant, _default_instructions, _greeting_message, is_inbound=False)

    else:
        # ── INBOUND CALL ──────────────────────────────────────────────────────
        logger.info(f"INBOUND CALL: Waiting for participant in room {ctx.room.name}")

        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error("Inbound: timed out waiting for participant - shutting down")
            ctx.shutdown()
            return

        logger.info(f"Inbound participant joined: {participant.identity}")
        await run_voice_agent(ctx, participant, _default_instructions, _inbound_greeting_message, is_inbound=True)


# ============ VOICE AGENT ============

async def run_voice_agent(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    instructions: str,
    greeting_message: str,
    is_inbound: bool = True,
):
    logger.info(f"INITIALIZING VOICE AGENT (is_inbound={is_inbound})")

    stt_instance = sarvam.STT(
        language="unknown",
        model="saarika:v2.5"
    )

    llm_instance = openai.LLM(
        model="gpt-4",
        temperature=0.7,
    )

    tts_instance = openai.TTS(
        voice="alloy",
        speed=1.0,
    )

    stable_user_id = participant.identity or ctx.job.metadata or "unknown_user"

    assistant = MultilingualAgent(
        user_id=stable_user_id,
        instructions=instructions,
    )
    assistant._stt_instance = stt_instance

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
    )

    lf_trace_id = None
    if langfuse_client and LANGFUSE_AVAILABLE:
        try:
            lf_trace_id = langfuse_client.create_trace_id()
            logger.info(f"Langfuse trace created: {lf_trace_id}")
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")

    conversation_buffer = {
        "last_user_input": None,
        "last_agent_response": None,
    }

    # ── ALL EVENT HANDLERS BEFORE session.start() ──────────────────────────

    @session.on("user_speech_committed")
    def on_user_speech_committed(event):
        try:
            if hasattr(event, 'alternatives') and event.alternatives:
                transcript = event.alternatives[0].text
                language = getattr(event.alternatives[0], 'language', 'unknown')
                stt_logger.info(f"Transcript: {transcript} | Language: {language}")

                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        langfuse_client.generation(
                            trace_id=lf_trace_id,
                            name="user_speech",
                            input=transcript,
                            metadata={"stt_language": language},
                        )
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")

            elif hasattr(event, 'text'):
                stt_logger.info(f"USER SPEECH (Fallback): {event.text}")

        except Exception as e:
            stt_logger.error(f"Error in user_speech_committed: {e}")

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        vad_logger.info("User started speaking...")

    @session.on("agent_speech_committed")
    def on_agent_speech_committed(event):
        try:
            agent_text = None
            if hasattr(event, 'text'):
                agent_text = event.text
            elif hasattr(event, 'alternatives') and event.alternatives:
                agent_text = event.alternatives[0].text

            if agent_text:
                logger.info(f"AGENT RESPONSE: {agent_text}")
                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        langfuse_client.generation(
                            trace_id=lf_trace_id,
                            name="agent_response",
                            output=agent_text,
                        )
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        logger.info(
            f"User input transcribed: {event.transcript}, "
            f"language: {event.language}, "
            f"final: {event.is_final}, "
            f"speaker_id: {event.speaker_id}"
        )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        try:
            role = event.item.role
            text_content = event.item.text_content
            interrupted = event.item.interrupted

            logger.debug(f"Conversation [{role}]: {text_content} | interrupted={interrupted}")

            for content in event.item.content:
                if isinstance(content, str):
                    if role == "user" and any(
                        phrase in content.lower()
                        for phrase in [
                            "my name is", "my dog's name is", "i prefer",
                            "remember that", "i am preparing for",
                        ]
                    ):
                        store_user_memory(user_id=assistant.user_id, text=content)

                    if role == "user":
                        logger.info(f"LLM INPUT (User): {content}")
                        conversation_buffer["last_user_input"] = content

                    elif role == "assistant":
                        logger.info(f"LLM OUTPUT (Agent): {content}")
                        if conversation_buffer["last_user_input"]:
                            asyncio.create_task(
                                log_llm_interaction(
                                    user_input=conversation_buffer["last_user_input"],
                                    llm_response=content,
                                    session_metadata={
                                        "room": ctx.room.name,
                                        "interrupted": interrupted,
                                    }
                                )
                            )
                            conversation_buffer["last_user_input"] = None
                            conversation_buffer["last_agent_response"] = None

                elif isinstance(content, ImageContent):
                    logger.debug(f"Image content: {content.image}")
                elif isinstance(content, AudioContent):
                    logger.debug(f"Audio content transcript: {content.transcript}")

        except Exception as e:
            logger.error(f"Error in conversation_item_added: {e}")
            import traceback
            traceback.print_exc()

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        logger.info("metrics_collected event fired")
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ── SESSION START ──────────────────────────────────────────────────────
    try:
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                # INBOUND:  close_on_disconnect=False
                #   The session itself stays alive briefly after hangup so the
                #   agent can finish any in-flight TTS/LLM work gracefully.
                #   We use the participant_disconnected event below to call
                #   ctx.shutdown() ourselves, which releases the job slot.
                #
                # OUTBOUND: close_on_disconnect=True
                #   Automatically tears everything down the moment the callee
                #   hangs up, which is the normal outbound behaviour.
                close_on_disconnect=not is_inbound,
            ),
        )
        logger.info("Agent session started successfully")

        await asyncio.sleep(1.5)

        logger.info(f"SENDING GREETING: {greeting_message}")
        await session.say(greeting_message, allow_interruptions=True)
        logger.info("Greeting sent successfully")

        # ── INBOUND ONLY: wait for caller to hang up then free the job slot ──
        # Without this the entrypoint coroutine never returns and the worker
        # slot stays occupied forever, causing the "3rd call fails" pattern.
        if is_inbound:
            disconnect_event = asyncio.Event()

            @ctx.room.on("participant_disconnected")
            def on_participant_disconnected(p: rtc.RemoteParticipant):
                if p.identity == participant.identity:
                    logger.info(f"Caller {p.identity} disconnected — releasing job slot")
                    disconnect_event.set()

            await disconnect_event.wait()
            logger.info("Inbound call ended — shutting down job cleanly")
            ctx.shutdown()   # ← frees the slot so next inbound call can use it

    except Exception as e:
        logger.error(f"Error in voice agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if langfuse_client and LANGFUSE_AVAILABLE:
            try:
                langfuse_client.flush()
                logger.info("Langfuse events flushed")
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")


# ============ PREWARM ============

def prewarm(proc: JobProcess):
    logger.info("PREWARMING VAD MODEL")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model prewarmed successfully")


# ============ API SERVER THREAD ============

def run_api_server_thread(host: str = "0.0.0.0", port: int = 8000):
    logger.info(f"STARTING API SERVER on {host}:{port}")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


# ============ MAIN ============
#
# Run:  python app.py start    ← production (Azure)
#       python app.py dev      ← local development
#
# Worker stays permanently registered with LiveKit.
# Inbound calls work at any time — no outbound call needed first.
# Outbound calls triggered via POST /dispatch/call.

if __name__ == "__main__":
    from datetime import datetime

    agent_worker_status["running"] = True
    agent_worker_status["started_at"] = datetime.now().isoformat()

    api_port = int(os.getenv("API_PORT", "8000"))
    api_host = os.getenv("API_HOST", "0.0.0.0")

    # FastAPI runs in a background daemon thread
    api_thread = threading.Thread(
        target=run_api_server_thread,
        args=(api_host, api_port),
        daemon=True,
        name="APIServer"
    )
    api_thread.start()
    logger.info(f"API server thread started on {api_host}:{api_port}")

    logger.info("Starting persistent LiveKit agent worker...")

    # cli.run_app blocks forever — worker stays registered permanently.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
            prewarm_fnc=prewarm,
        )
    )