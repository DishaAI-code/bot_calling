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
    Agent,
    AgentSession,
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import openai, silero, azure
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import threading
import uvicorn

# Langfuse SDK for better tracing
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

# load environment variables
load_dotenv(dotenv_path=".env")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
azure_speech_region = os.getenv("AZURE_SPEECH_REGION")

# Global Langfuse client
langfuse_client = None
langfuse_trace = None

# Global agent worker status
agent_worker_status = {
    "running": False,
    "started_at": None,
    "calls_dispatched": 0,
}

# Initialize FastAPI app
app = FastAPI(
    title="LiveKit Outbound Caller API",
    description="API endpoints for managing LiveKit outbound calls",
    version="1.0.0"
)

# Pydantic models for API requests
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
    """Root endpoint with API information"""
    return {
        "service": "LiveKit Outbound Caller API",
        "version": "1.0.0",
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_worker": "running" if agent_worker_status["running"] else "stopped",
        "timestamp": perf_counter()
    }

@app.get("/status")
async def get_status():
    """Get agent worker status and statistics"""
    return {
        "agent_worker": agent_worker_status,
        "environment": {
            "livekit_url": os.getenv("LIVEKIT_URL", "not set"),
            "sip_trunk_configured": bool(outbound_trunk_id and outbound_trunk_id.startswith("ST_")),
            "azure_stt_configured": bool(azure_speech_key and azure_speech_region)
        }
    }

@app.post("/dispatch/call", response_model=CallResponse)
async def dispatch_call(request: CallRequest):
    """
    Dispatch a single outbound call
    
    Args:
        phone_number: Phone number to call (e.g., "+1234567890")
        room_name: Optional custom room name (auto-generated if not provided)
    
    Returns:
        CallResponse with dispatch status
    """
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
        
        # Create LiveKit API client
        lk_api = api.LiveKitAPI(
            url=livekit_url,
            api_key=livekit_api_key,
            api_secret=livekit_api_secret,
        )
        
        # Generate room name if not provided
        room_name = request.room_name or f"call-{request.phone_number.replace('+', '')}-{int(perf_counter() * 1000)}"
        
        # Create room
        await lk_api.room.create_room(
            api.CreateRoomRequest(name=room_name)
        )
        
        logger.info(f"Created room: {room_name}")
        
        # Dispatch agent to room
        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="outbound-caller",
                metadata=request.phone_number,
            )
        )
        
        logger.info(f"Dispatched call to {request.phone_number} in room {room_name}")
        
        # Update statistics
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dispatch call: {str(e)}"
        )

@app.post("/dispatch/batch")
async def dispatch_batch_calls(request: BatchCallRequest):
    """
    Dispatch multiple outbound calls
    
    Args:
        phone_numbers: List of phone numbers to call
    
    Returns:
        Summary of dispatch results
    """
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
            
            # Small delay between calls
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
def setup_langfuse(
    host: str | None = None, 
    public_key: str | None = None, 
    secret_key: str | None = None
):
    """Set up Langfuse tracing for the LiveKit agent"""
    global langfuse_client
    
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_BASE_URL")

    if not public_key or not secret_key or not host:
        logger.warning("Langfuse not configured - tracing disabled. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL to enable.")
        return False

    try:
        # Set up OpenTelemetry tracing
        langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        set_tracer_provider(trace_provider)
        
        # Also initialize Langfuse SDK for better input/output logging
        if LANGFUSE_AVAILABLE:
            langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info("Langfuse tracing enabled (OpenTelemetry + SDK)")
        else:
            logger.info("Langfuse tracing enabled (OpenTelemetry only)")
            logger.info("Install 'langfuse' package for better input/output logging")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up Langfuse tracing: {e}")
        return False


# System prompt for multilingual AI voice assistant
_default_instructions = """You are a multilingual AI voice assistant calling from Autodesk. Follow these rules strictly:

CORE BEHAVIOR:
1. You are making an outbound call to assist with Autodesk products and services.
2. The greeting has already been said - DO NOT repeat it.
3. Keep responses SHORT (1-3 sentences), natural, and conversational for phone calls.
4. Be helpful, friendly, and professional.

CRITICAL LANGUAGE DETECTION:
5. **DETECT the user's language from their ACTUAL WORDS, not the language tag.**
6. The speech-to-text system may make mistakes - YOU must figure out the true language.
7. Look at the actual vocabulary, grammar, and script to determine language:
   - English: "hello", "what", "how", "thank you", ASCII text
   - Hindi: "नमस्ते", "क्या", "कैसे", Devanagari script OR romanized ("kya", "kaise", "mujhe")
   - Gujarati: "નમસ્તે", Gujarati script, unique Gujarati vocabulary
   - Kannada: "ನಮಸ್ಕಾರ", Kannada script, unique Kannada vocabulary

LANGUAGE RESPONSE RULES:
8. **ALWAYS respond in the SAME language as the user's message.**
9. If you see Kannada/Hindi/Gujarati script for what sounds like English words (e.g., "ಹಲೋ" for "hello"), this is a transcription error:
   - Ask the user in English: "I'm hearing you, but having trouble with the transcription. Could you repeat that please?"
   - Try to understand the intent and respond in English
10. If the user clearly speaks Hindi → Respond in Hindi (Devanagari: हिंदी में)
11. If the user clearly speaks Gujarati → Respond in Gujarati (ગુજરાતીમાં)
12. If the user clearly speaks Kannada → Respond in Kannada (ಕನ್ನಡದಲ್ಲಿ)
13. If the user clearly speaks English → Respond in English
14. When user switches languages mid-conversation → Immediately switch with them

CONVERSATION MANAGEMENT:
15. Answer questions about Autodesk products, software, and services in the user's language.
16. If you don't know something, be honest and offer to help differently.
17. Allow the user to end the conversation naturally - don't be pushy.
18. Be patient with accents and transcription errors - focus on understanding intent.
19. This is a PHONE CALL - speak naturally, not like written text.

HANDLING TRANSCRIPTION ERRORS:
20. If you see mixed scripts or gibberish, politely ask the user to repeat.
21. Focus on understanding the user's INTENT, not perfect transcription.
22. If unsure about language, default to English and let the user guide you.

IMPORTANT: Your primary job is to UNDERSTAND and HELP the user, regardless of transcription quality."""

_greeting_message = "Hello! I'm your AI assistant calling from Autodesk. I can speak English, Hindi, Gujarati, and Kannada. How may I help you today?"


# ============ MAIN AGENT CODE ============

async def entrypoint(ctx: JobContext):
    global _default_instructions, _greeting_message, outbound_trunk_id
    
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Set up Langfuse tracing after connection (reduces init time)
    setup_langfuse()

    user_identity = "phone_user"
    phone_number = ctx.job.metadata
    logger.info(f"dialing {phone_number} to room {ctx.room.name}")

    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )

    participant = await ctx.wait_for_participant(identity=user_identity)
    await run_voice_agent(ctx, participant, _default_instructions, _greeting_message)

    # monitor the call status
    start_time = perf_counter()
    while perf_counter() - start_time < 30:
        call_status = participant.attributes.get("sip.callStatus")
        if call_status == "active":
            logger.info("user has picked up")
            return
        elif call_status == "automation":
            pass
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
            logger.info("user rejected the call, exiting job")
            break
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
            logger.info("user did not pick up, exiting job")
            break
        await asyncio.sleep(0.1)

    logger.info("session timed out, exiting job")
    ctx.shutdown()


class VoiceAssistant(Agent):
    """Voice assistant with multilingual support"""
    
    def __init__(self, instructions: str, api_client: api.LiveKitAPI, room: rtc.Room):
        super().__init__(instructions=instructions)
        self.api = api_client
        self.room = room

    async def hangup(self, participant_identity: str):
        """Helper method to end the call"""
        try:
            await self.api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=self.room.name,
                    identity=participant_identity,
                )
            )
        except Exception as e:
            logger.info(f"received error while ending call: {e}")

    @function_tool()
    async def end_call(self, context: RunContext) -> str:
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {context.participant.identity}")
        await self.hangup(context.participant.identity)
        return "Call ended"

    @function_tool()
    async def detected_answering_machine(self, context: RunContext) -> str:
        """Called when the call reaches voicemail"""
        logger.info(f"detected answering machine for {context.participant.identity}")
        await self.hangup(context.participant.identity)
        return "Voicemail detected, call ended"


async def run_voice_agent(
    ctx: JobContext, 
    participant: rtc.RemoteParticipant, 
    instructions: str,
    greeting_message: str
):
    logger.info("🚀 Starting voice agent with Azure Multilingual STT")

    assistant = VoiceAssistant(
        instructions=instructions,
        api_client=ctx.api,
        room=ctx.room,
    )

    # ============ AZURE STT CONFIGURATION ============
    # Multilingual auto-detection - Hindi first for Indian speakers
    # Azure will automatically detect the language from the candidate set
    
    azure_stt = azure.STT(
        speech_key=azure_speech_key,
        speech_region=azure_speech_region,
        language=["hi-IN", "en-IN", "gu-IN", "kn-IN"],  # Hindi FIRST, then English, then other Indian languages
        sample_rate=16000,
        num_channels=1,
    )
    logger.info("✅ Azure STT: Multilingual auto-detection")
    logger.info("   Languages: hi-IN, en-IN, gu-IN, kn-IN (Hindi prioritized)")
    logger.info("   Azure will auto-detect language from speech")

    # Use GPT-4 for better multilingual understanding and response
    llm = openai.LLM(
        model="gpt-4",
        temperature=0.7,  # Slightly creative for natural conversation
    )

    # Use OpenAI TTS - it has good multilingual support
    # The 'alloy' voice works well for multiple languages
    tts = openai.TTS(
        voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
        speed=1.0,
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=azure_stt,
        llm=llm,
        tts=tts,
    )

    # Initialize Langfuse trace ID for this conversation
    lf_trace_id = None
    if langfuse_client and LANGFUSE_AVAILABLE:
        try:
            lf_trace_id = langfuse_client.create_trace_id()
            logger.info(f" Langfuse trace created: {lf_trace_id}")
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")

    # Event listeners for debugging and Langfuse logging
    @session.on("user_speech_committed")
    def on_user_speech(event):
        try:
            if event.alternatives and len(event.alternatives) > 0:
                transcript = event.alternatives[0].text
                language = getattr(event.alternatives[0], 'language', 'unknown')
                
                # Detect script/character type to identify actual language
                has_kannada = any('\u0C80' <= char <= '\u0CFF' for char in transcript)
                has_hindi = any('\u0900' <= char <= '\u097F' for char in transcript)
                has_gujarati = any('\u0A80' <= char <= '\u0AFF' for char in transcript)
                is_ascii = transcript.isascii()
                
                # Determine actual language from content
                actual_language = "Unknown"
                if is_ascii:
                    # Could be English or Romanized Indian language
                    hindi_keywords = ['kya', 'hai', 'mujhe', 'chahiye', 'kaise', 'aap', 'main']
                    if any(word in transcript.lower() for word in hindi_keywords):
                        actual_language = "Hindi (Romanized)"
                    else:
                        actual_language = "English"
                elif has_kannada:
                    actual_language = "Kannada"
                elif has_hindi:
                    actual_language = "Hindi"
                elif has_gujarati:
                    actual_language = "Gujarati"
                
                # Console logging with better formatting
                logger.info("=" * 80)
                logger.info(f"👤 USER SPEECH DETECTED")
                logger.info(f"   Azure STT Language Tag: {language}")
                logger.info(f"   Detected Language (Content): {actual_language}")
                logger.info(f"   Transcript: {transcript}")
                
                # Warn about potential transcription mismatches
                if language == 'kn-IN' and is_ascii:
                    logger.warning(f"   ⚠️  MISMATCH: Azure tagged as Kannada but content is ASCII")
                    logger.warning(f"   💡 Likely English misdetected - GPT-4 will interpret correctly")
                elif language == 'en-US' and (has_kannada or has_hindi or has_gujarati):
                    logger.warning(f"   ⚠️  MISMATCH: Azure tagged as English but content has Indian script")
                    logger.warning(f"   💡 GPT-4 will respond in the correct language")
                
                # Show confidence that GPT-4 will handle it
                if language != actual_language:
                    logger.info(f"   ✅ GPT-4 will analyze content and respond appropriately")
                
                logger.info("=" * 80)
                
                # Log to Langfuse with proper input field
                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        gen = langfuse_client.start_generation(
                            trace_id=lf_trace_id,
                            name="user_speech_to_text",
                            input=transcript,
                            metadata={
                                "azure_language_tag": language,
                                "actual_detected_language": actual_language,
                                "has_kannada_script": has_kannada,
                                "has_hindi_script": has_hindi,
                                "has_gujarati_script": has_gujarati,
                                "is_ascii": is_ascii,
                                "language_mismatch": language != actual_language,
                            }
                        )
                        gen.end()
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")
                        
        except Exception as e:
            logger.warning(f"Error logging user speech: {e}")

    @session.on("agent_speech_committed")
    def on_agent_speech(event):
        try:
            if hasattr(event, 'text'):
                agent_text = event.text
                
                # Detect response language/script
                has_kannada = any('\u0C80' <= char <= '\u0CFF' for char in agent_text)
                has_hindi = any('\u0900' <= char <= '\u097F' for char in agent_text)
                has_gujarati = any('\u0A80' <= char <= '\u0AFF' for char in agent_text)
                is_english = agent_text.isascii()
                
                logger.info("=" * 80)
                logger.info(f"🤖 AGENT RESPONSE")
                logger.info(f"   Text: {agent_text[:150]}{'...' if len(agent_text) > 150 else ''}")
                
                # Show what language the agent responded in
                if is_english:
                    logger.info(f"   ✅ Response Language: English")
                elif has_kannada:
                    logger.info(f"   Response Language: Kannada")
                elif has_hindi:
                    logger.info(f"   Response Language: Hindi")
                elif has_gujarati:
                    logger.info(f"   Response Language: Gujarati")
                
                logger.info("=" * 80)
                
                # Log to Langfuse with proper output field
                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        gen = langfuse_client.start_generation(
                            trace_id=lf_trace_id,
                            name="agent_text_response",
                            output=agent_text,
                            metadata={
                                "response_length": len(agent_text),
                                "is_english": is_english,
                                "has_kannada": has_kannada,
                                "has_hindi": has_hindi,
                                "has_gujarati": has_gujarati,
                            }
                        )
                        gen.end()
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")
                        
        except Exception as e:
            logger.warning(f"Error logging agent speech: {e}")

    try:
        # Start the session
        await session.start(room=ctx.room, agent=assistant)
        
        # Wait for session to be ready
        await asyncio.sleep(1)
        
        # Send greeting message
        logger.info("📞 Sending greeting message")
        await session.say(greeting_message, allow_interruptions=True)
    finally:
        # Flush Langfuse events before exiting
        if langfuse_client and LANGFUSE_AVAILABLE:
            try:
                langfuse_client.flush()
                logger.info("Langfuse events flushed")
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def auto_dispatch_calls():
    """Automatically dispatch calls from a queue or configuration"""
    
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        raise ValueError("LiveKit credentials not set in environment")
    
    lk_api = api.LiveKitAPI(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )
    
    phone_numbers_file = "phone_numbers.txt"
    
    if os.path.exists(phone_numbers_file):
        with open(phone_numbers_file, "r") as f:
            phone_numbers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        phone_numbers = ["+916203879448"]
        logger.info(f"No phone_numbers.txt found, using default: {phone_numbers[0]}")
    
    logger.info(f"Starting auto-dispatch for {len(phone_numbers)} phone number(s)")
    
    for phone_number in phone_numbers:
        try:
            logger.info(f"Dispatching call to {phone_number}")
            
            room_name = f"call-{phone_number.replace('+', '')}-{int(perf_counter() * 1000)}"
            
            await lk_api.room.create_room(
                api.CreateRoomRequest(name=room_name)
            )
            
            logger.info(f"Created room: {room_name}")
            
            dispatch = await lk_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=room_name,
                    agent_name="outbound-caller",
                    metadata=phone_number,
                )
            )
            
            logger.info(f"Dispatch created successfully for {phone_number}")
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error dispatching call to {phone_number}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("All calls dispatched")
    await lk_api.aclose()


def run_agent_worker():
    """Run the LiveKit agent worker (must be in main thread for signal handling)"""
    import sys
    from datetime import datetime
    
    logger.info("Starting LiveKit Agent Worker in main thread...")
    agent_worker_status["running"] = True
    agent_worker_status["started_at"] = datetime.now().isoformat()
    
    try:
        # Clear sys.argv to prevent LiveKit CLI from parsing our custom arguments
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], "dev"]  # Run in dev mode for the worker
        
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="outbound-caller",
                prewarm_fnc=prewarm,
            )
        )
        
        # Restore original argv
        sys.argv = original_argv
    except Exception as e:
        logger.error(f"Agent worker error: {e}")
        agent_worker_status["running"] = False
        raise


def run_api_server_thread(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in a background thread"""
    import uvicorn
    
    logger.info(f"Starting API Server on {host}:{port}")
    logger.info(f"API Documentation available at http://{host}:{port}/docs")
    logger.info(f"Health Check: http://{host}:{port}/health")
    logger.info(f"Dispatch Call: POST http://{host}:{port}/dispatch/call")
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()


def run_api_mode():
    """Run both agent worker and API server together"""
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError("SIP_OUTBOUND_TRUNK_ID is not set")
    
    logger.info("=" * 80)
    logger.info("Starting LiveKit Outbound Caller - API Mode")
    logger.info("=" * 80)
    logger.info("This will start:")
    logger.info("  1. FastAPI Server (background thread)")
    logger.info("  2. LiveKit Agent Worker (main thread)")
    logger.info("=" * 80)
    
    # Get API configuration
    api_port = int(os.getenv("API_PORT", "8000"))
    api_host = os.getenv("API_HOST", "0.0.0.0")
    
    # Start API server in a background thread (doesn't need signal handling)
    logger.info("Starting API server in background thread...")
    api_thread = threading.Thread(
        target=run_api_server_thread,
        args=(api_host, api_port),
        daemon=True,
        name="APIServer"
    )
    api_thread.start()
    
    # Give the API server a moment to start
    import time
    logger.info("Waiting for API server to initialize...")
    time.sleep(2)
    
    if api_thread.is_alive():
        logger.info("API server thread is running!")
    else:
        logger.error("API server thread failed to start!")
    
    logger.info("=" * 80)
    
    # Start agent worker in main thread (needs signal handling for LiveKit CLI)
    run_agent_worker()


if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError("SIP_OUTBOUND_TRUNK_ID is not set")
    
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "api":
            # New API mode - runs both agent worker and API server
            run_api_mode()
        
        elif mode == "auto":
            # Legacy auto-dispatch mode
            logger.info("Running in auto-dispatch mode")
            asyncio.run(auto_dispatch_calls())
        
        elif mode == "dev":
            # Development mode - just agent worker
            logger.info("Running in development mode (agent worker only)")
            cli.run_app(
                WorkerOptions(
                    entrypoint_fnc=entrypoint,
                    agent_name="outbound-caller",
                    prewarm_fnc=prewarm,
                )
            )
        else:
            print("Usage:")
            print("  python agent.py api   - Run API server with agent worker (for production)")
            print("  python agent.py dev   - Run agent worker only (for development)")
            print("  python agent.py auto  - Auto-dispatch calls from phone_numbers.txt")
    else:
        # Default to API mode for container deployment
        logger.info("No mode specified, defaulting to API mode")
        run_api_mode()

