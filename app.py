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
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import openai, silero, deepgram
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, cast
import threading
import uvicorn

# For Whisper on Groq
from groq import Groq

# Langfuse SDK for better tracing
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Enhanced logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.DEBUG)

# Create separate loggers for different components
stt_logger = logging.getLogger("STT")
stt_logger.setLevel(logging.DEBUG)

lang_detect_logger = logging.getLogger("LanguageDetection")
lang_detect_logger.setLevel(logging.DEBUG)

vad_logger = logging.getLogger("VAD")
vad_logger.setLevel(logging.DEBUG)

# Environment variables
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate critical API keys
if not deepgram_api_key:
    logger.error("❌ DEEPGRAM_API_KEY not set in environment!")
if not groq_api_key:
    logger.error("❌ GROQ_API_KEY not set in environment!")

# Global Langfuse client
langfuse_client = None

# Global agent worker status
agent_worker_status = {
    "running": False,
    "started_at": None,
    "calls_dispatched": 0,
}

# Language detection statistics
language_stats = {
    "detections": [],
    "switches": 0,
    "current_language": "en-US",
}

# Initialize FastAPI app
app = FastAPI(
    title="LiveKit Outbound Caller API",
    description="API endpoints for managing LiveKit outbound calls with multilingual support",
    version="2.1.0-fixed"
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
        "service": "LiveKit Outbound Caller API - Fixed Dual STT",
        "version": "2.1.0",
        "status": "running",
        "features": ["Deepgram Primary STT", "Groq Whisper Language Detection", "Proper Event Logging"],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "language_stats": "/language-stats",
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
        "deepgram_configured": bool(deepgram_api_key),
        "groq_configured": bool(groq_api_key),
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
            "deepgram_configured": bool(deepgram_api_key),
            "groq_configured": bool(groq_api_key),
        }
    }

@app.get("/language-stats")
async def get_language_stats():
    """Get language detection statistics"""
    return {
        "current_language": language_stats["current_language"],
        "total_detections": len(language_stats["detections"]),
        "language_switches": language_stats["switches"],
        "detection_history": language_stats["detections"][-10:],
    }

@app.post("/dispatch/call", response_model=CallResponse)
async def dispatch_call(request: CallRequest):
    """Dispatch a single outbound call"""
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
        
        await lk_api.room.create_room(
            api.CreateRoomRequest(name=room_name)
        )
        
        logger.info(f"✅ Created room: {room_name}")
        
        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="outbound-caller",
                metadata=request.phone_number,
            )
        )
        
        logger.info(f"✅ Dispatched call to {request.phone_number} in room {room_name}")
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
        logger.error(f"❌ Error dispatching call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dispatch call: {str(e)}"
        )

@app.post("/dispatch/batch")
async def dispatch_batch_calls(request: BatchCallRequest):
    """Dispatch multiple outbound calls"""
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
            logger.error(f"❌ Failed to dispatch call to {phone_number}: {e}")
    
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
    host = host or os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not public_key or not secret_key or not host:
        logger.warning("⚠️  Langfuse not configured - tracing disabled")
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
            logger.info("✅ Langfuse tracing enabled (OpenTelemetry + SDK)")
        else:
            logger.info("✅ Langfuse tracing enabled (OpenTelemetry only)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to set up Langfuse tracing: {e}")
        return False


# ============ LANGUAGE DETECTION WITH WHISPER (GROQ) ============

class LanguageDetector:
    """Handles language detection using Whisper on Groq"""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.current_language = "en"
        self.detection_count = 0
        lang_detect_logger.info("✅ LanguageDetector initialized with Groq Whisper")
    
    async def detect_language_from_frames(self, audio_frames: list) -> Optional[str]:
        """
        Detect language from audio frames using Whisper
        Returns ISO 639-1 language code (e.g., 'en', 'hi', 'gu', 'kn')
        """
        try:
            self.detection_count += 1
            lang_detect_logger.info("=" * 80)
            lang_detect_logger.info(f"🔍 LANGUAGE DETECTION #{self.detection_count} STARTED")
            lang_detect_logger.info(f"📊 Processing {len(audio_frames)} audio frames")
            
            # Combine audio frames into a single buffer
            combined = rtc.combine_audio_frames(audio_frames)
            wav_data = combined.to_wav_bytes()
            
            lang_detect_logger.info(f"📦 Audio data size: {len(wav_data)} bytes ({len(wav_data)/1024:.2f} KB)")
            
            # Call Whisper API on Groq
            start_time = perf_counter()
            lang_detect_logger.info("🚀 Calling Groq Whisper API...")
            
            response = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                file=("audio.wav", wav_data, "audio/wav"),
                model="whisper-large-v3",
                response_format="verbose_json",
                temperature=0.0,
            )
            
            elapsed = perf_counter() - start_time
            lang_detect_logger.info(f"⏱️  Whisper API response time: {elapsed:.3f}s")
            
            # Extract language from response
            detected_language = getattr(response, 'language', None)
            transcript = getattr(response, 'text', '')
            
            if detected_language:
                # Map language names to ISO codes if needed
                lang_code = self._normalize_language_code(detected_language)
                
                lang_detect_logger.info("=" * 80)
                lang_detect_logger.info(f"🎯 WHISPER LANGUAGE DETECTED: {detected_language} ({lang_code})")
                lang_detect_logger.info(f"📝 Whisper Transcript: {transcript}")
                lang_detect_logger.info(f"⏱️  Total detection time: {elapsed:.3f}s")
                lang_detect_logger.info("=" * 80)
                
                # Update global stats
                language_stats["detections"].append({
                    "timestamp": perf_counter(),
                    "language": lang_code,
                    "language_full": detected_language,
                    "transcript": transcript,
                    "duration": elapsed,
                })
                
                return lang_code
            else:
                lang_detect_logger.warning(f"⚠️  No language detected in Whisper response")
                return None
                
        except Exception as e:
            lang_detect_logger.error(f"❌ Error detecting language: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_language_code(self, lang: str) -> str:
        """Normalize language code to ISO 639-1"""
        lang_map = {
            'english': 'en',
            'hindi': 'hi',
            'gujarati': 'gu',
            'kannada': 'kn',
            'tamil': 'ta',
            'telugu': 'te',
            'malayalam': 'ml',
            'bengali': 'bn',
            'marathi': 'mr',
            'punjabi': 'pa',
        }
        
        lang_lower = lang.lower()
        return lang_map.get(lang_lower, lang_lower[:2])


# ============ CUSTOM AGENT WITH LANGUAGE DETECTION ============

class MultilingualAgent(Agent):
    """
    Extended Agent with multilingual language detection support
    """
    
    def __init__(self, language_detector: LanguageDetector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_detector = language_detector
        self.current_stt_language = "en-US"  # Start with English
        self._vad_frames_buffer = []
        vad_logger.info("✅ MultilingualAgent initialized")
    
    async def handle_vad_event(self, ev: vad.VADEvent):
        """Handle VAD events for language detection"""
        try:
            vad_logger.info("=" * 80)
            vad_logger.info(f"🎤 VAD EVENT: User stopped speaking")
            vad_logger.info(f"⏱️  Duration: {ev.duration:.2f}s")
            vad_logger.info(f"📊 Frames count: {len(ev.frames)}")
            vad_logger.info(f"🔊 Speech probability: {ev.probability:.2%}")
            vad_logger.info("=" * 80)
            
            # Skip very short utterances
            if ev.duration < 0.8:
                vad_logger.info(f"⚠️  Utterance too short ({ev.duration:.2f}s), skipping language detection")
                return
            
            # Start language detection
            asyncio.create_task(self._detect_language(ev.frames))
            
        except Exception as e:
            vad_logger.error(f"❌ Error handling VAD event: {e}")
    
    async def _detect_language(self, audio_frames: list):
        """Detect language and update STT"""
        try:
            lang_detect_logger.info("=" * 80)
            lang_detect_logger.info(f"🔍 STARTING LANGUAGE DETECTION")
            lang_detect_logger.info(f"⏱️  Processing {len(audio_frames)} audio frames")
            lang_detect_logger.info("=" * 80)
            
            # Detect language using Whisper
            detected_lang = await self.language_detector.detect_language_from_frames(audio_frames)
            
            if detected_lang:
                # Get current language (strip region code if present)
                current_lang = self.current_stt_language.split('-')[0]
                
                if detected_lang != current_lang:
                    lang_detect_logger.info("=" * 80)
                    lang_detect_logger.info(f"🔀 LANGUAGE SWITCH DETECTED!")
                    lang_detect_logger.info(f"   FROM: {self.current_stt_language}")
                    lang_detect_logger.info(f"   TO: {detected_lang}")
                    lang_detect_logger.info("=" * 80)
                    
                    # Update statistics
                    language_stats["switches"] += 1
                    language_stats["current_language"] = detected_lang
                    
                    # Map to Deepgram language codes
                    deepgram_lang_map = {
                        'en': 'en-US',
                        'hi': 'hi',
                        'gu': 'hi',  # Deepgram treats Gujarati as Hindi
                        'kn': 'kn',
                        'ta': 'ta',
                        'te': 'te',
                        'ml': 'ml',
                        'bn': 'bn',
                        'mr': 'mr',
                    }
                    
                    new_deepgram_lang = deepgram_lang_map.get(detected_lang, 'en-US')
                    lang_detect_logger.info(f"📍 Language mapping: {detected_lang} → {new_deepgram_lang} (Deepgram format)")
                    
                    # Update Deepgram STT
                    self.current_stt_language = new_deepgram_lang
                    
                    stt_logger.info("=" * 80)
                    stt_logger.info(f"✅ LANGUAGE SWITCHED")
                    stt_logger.info(f"   New language: {new_deepgram_lang}")
                    stt_logger.info("=" * 80)
                    
        except Exception as e:
            lang_detect_logger.error(f"❌ Error in language detection: {e}")
            import traceback
            traceback.print_exc()


# ============ SYSTEM PROMPTS ============

_default_instructions = """You are a multilingual AI voice assistant for Autodesk. Follow these rules strictly:

LANGUAGE DETECTION:
1. The system automatically detects the language you're speaking using advanced speech recognition.
2. You will ALWAYS respond in the EXACT SAME LANGUAGE as the user's speech.
3. When the user switches language mid-conversation, immediately switch to that language.

SUPPORTED LANGUAGES:
- English
- Hindi (हिंदी)
- Gujarati (ગુજરાતી)
- Kannada (ಕನ್ನಡ)

CONVERSATION STYLE:
4. Keep responses short (1-3 sentences), friendly, and conversational.
5. Do not repeat the welcome greeting - it has already been said.
6. You are calling from Autodesk to assist the user with product information and queries.
7. Allow the user to end the conversation naturally.
8. Use natural, colloquial expressions appropriate for phone conversations in each language.

IMPORTANT: Respond in the user's detected language. The system has already identified it for you."""

_greeting_message = "Hello, I am your AI assistant calling from Autodesk. I can understand and speak English, Hindi, Gujarati, and Kannada. How can I help you today?"


# ============ MAIN AGENT CODE ============

async def entrypoint(ctx: JobContext):
    global _default_instructions, _greeting_message, outbound_trunk_id
    
    logger.info("=" * 80)
    logger.info("🚀 AGENT ENTRYPOINT STARTED")
    logger.info("=" * 80)
    
    # Set up Langfuse tracing
    setup_langfuse()
    
    logger.info(f"📞 Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_identity = "phone_user"
    phone_number = ctx.job.metadata
    logger.info(f"☎️  Dialing {phone_number} to room {ctx.room.name}")

    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )

    participant = await ctx.wait_for_participant(identity=user_identity)
    logger.info(f"✅ Participant joined: {user_identity}")
    
    await run_voice_agent(ctx, participant, _default_instructions, _greeting_message)

    # Monitor call status
    start_time = perf_counter()
    logger.info("⏳ Monitoring call status...")
    
    while perf_counter() - start_time < 30:
        call_status = participant.attributes.get("sip.callStatus")
        
        if call_status == "active":
            logger.info("✅ User has picked up - call is active")
            return
        elif call_status == "automation":
            logger.debug("🤖 Call status: automation")
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
            logger.info("❌ User rejected the call, exiting job")
            break
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
            logger.info("📵 User did not pick up, exiting job")
            break
            
        await asyncio.sleep(0.1)

    logger.info("⏱️  Session timed out, exiting job")
    ctx.shutdown()


async def run_voice_agent(
    ctx: JobContext, 
    participant: rtc.RemoteParticipant, 
    instructions: str,
    greeting_message: str
):
    logger.info("=" * 80)
    logger.info("🎙️  INITIALIZING VOICE AGENT WITH DUAL-STT")
    logger.info("=" * 80)
    
    # Validate API keys
    if not deepgram_api_key:
        raise ValueError("❌ DEEPGRAM_API_KEY is not set in environment")
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY is not set in environment")
    
    logger.info("✅ API Keys validated")
    
    # Initialize Language Detector (Whisper on Groq)
    logger.info("🔧 Initializing Language Detector (Groq Whisper)...")
    language_detector = LanguageDetector(groq_api_key)
    
    # Initialize Deepgram STT - START WITH ENGLISH DEFAULT!
    logger.info("🔧 Initializing Deepgram STT (Primary)...")
    stt_instance = deepgram.STT(
        model="nova-2",
        language="en-US",  # START WITH ENGLISH DEFAULT!
        api_key=deepgram_api_key,
    )
    stt_logger.info("✅ Deepgram STT initialized with language: en-US (default)")

    # Initialize LLM (GPT-4 for better multilingual understanding)
    logger.info("🔧 Initializing GPT-4 LLM...")
    llm_instance = openai.LLM(
        model="gpt-4",
        temperature=0.7,
    )
    logger.info("✅ GPT-4 LLM initialized")

    # Initialize TTS (OpenAI TTS with multilingual support)
    logger.info("🔧 Initializing OpenAI TTS...")
    tts_instance = openai.TTS(
        voice="alloy",
        speed=1.0,
    )
    logger.info("✅ OpenAI TTS initialized")

    # Create multilingual agent
    logger.info("🔧 Creating MultilingualAgent...")
    assistant = MultilingualAgent(
        language_detector=language_detector,
        instructions=instructions,
    )

    # Create agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
    )

    # Initialize Langfuse trace ID
    lf_trace_id = None
    if langfuse_client and LANGFUSE_AVAILABLE:
        try:
            lf_trace_id = langfuse_client.create_trace_id()
            logger.info(f"📊 Langfuse trace created: {lf_trace_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to create Langfuse trace: {e}")

    # ============ FIXED EVENT HANDLERS ============

    @session.on("user_speech_committed")
    def on_user_speech_committed(event):
        """Handle user speech with proper event structure - FIXED VERSION"""
        try:
            if hasattr(event, 'alternatives') and event.alternatives and len(event.alternatives) > 0:
                transcript = event.alternatives[0].text
                language = getattr(event.alternatives[0], 'language', 'unknown')
                
                stt_logger.info("=" * 80)
                stt_logger.info(f"👤 USER SPEECH DETECTED (Deepgram)")
                stt_logger.info(f"📝 Transcript: {transcript}")
                stt_logger.info(f"🌍 Language: {language}")
                stt_logger.info(f"🔧 Current STT Setting: {assistant.current_stt_language}")
                stt_logger.info("=" * 80)
                
                # Log to Langfuse
                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        langfuse_client.generation(
                            trace_id=lf_trace_id,
                            name="user_speech_deepgram",
                            input=transcript,
                            metadata={
                                "stt_language": language,
                                "current_stt_setting": assistant.current_stt_language,
                                "transcript_length": len(transcript),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")
                        
            elif hasattr(event, 'text'):
                # Fallback for different event structure
                transcript = event.text
                stt_logger.info("=" * 80)
                stt_logger.info(f"👤 USER SPEECH (Fallback): {transcript}")
                stt_logger.info("=" * 80)
                
        except Exception as e:
            stt_logger.error(f"❌ Error in user_speech_committed: {e}")
            import traceback
            traceback.print_exc()

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        """Log when user starts speaking"""
        vad_logger.info("🎤 User started speaking...")

    @session.on("user_stopped_speaking")
    def on_user_stopped_speaking(ev: vad.VADEvent):
        """Handle user stopped speaking for language detection"""
        asyncio.create_task(assistant.handle_vad_event(ev))

    @session.on("agent_speech_committed")
    def on_agent_speech_committed(event):
        """Handle agent speech responses - FIXED VERSION"""
        try:
            if hasattr(event, 'text'):
                agent_text = event.text
                logger.info("=" * 80)
                logger.info(f"🤖 AGENT RESPONSE:")
                logger.info(f"📝 {agent_text}")
                logger.info("=" * 80)
                
                # Log to Langfuse
                if langfuse_client and LANGFUSE_AVAILABLE and lf_trace_id:
                    try:
                        langfuse_client.generation(
                            trace_id=lf_trace_id,
                            name="agent_response",
                            output=agent_text,
                            metadata={
                                "response_length": len(agent_text),
                                "current_language": assistant.current_stt_language,
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Langfuse logging error: {e}")
                        
            elif hasattr(event, 'alternatives') and event.alternatives and len(event.alternatives) > 0:
                agent_text = event.alternatives[0].text
                logger.info("=" * 80)
                logger.info(f"🤖 AGENT RESPONSE:")
                logger.info(f"📝 {agent_text}")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"❌ Error in agent_speech_committed: {e}")
            import traceback
            traceback.print_exc()

    # Debug handler to see all events
    @session.on("any")
    def on_any_event(event_name, *args):
        """Debug all events - helps identify what's firing"""
        if event_name not in ["audio_stream"]:  # Skip noisy audio events
            logger.debug(f"🔔 EVENT FIRED: {event_name} - Args: {len(args)}")

    try:
        # Start the session
        await session.start(room=ctx.room, agent=assistant)
        logger.info("✅ Agent session started successfully")
        
        # Wait for session to be ready
        await asyncio.sleep(1.5)
        
        # Send greeting message
        logger.info("=" * 80)
        logger.info(f"📢 SENDING GREETING MESSAGE")
        logger.info(f"📝 Message: {greeting_message}")
        logger.info("=" * 80)
        await session.say(greeting_message, allow_interruptions=True)
        logger.info("✅ Greeting sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in voice agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Flush Langfuse events
        if langfuse_client and LANGFUSE_AVAILABLE:
            try:
                langfuse_client.flush()
                logger.info("✅ Langfuse events flushed")
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")


def prewarm(proc: JobProcess):
    """Prewarm VAD model"""
    logger.info("=" * 80)
    logger.info("🔥 PREWARMING VAD MODEL")
    logger.info("=" * 80)
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("✅ VAD model prewarmed successfully")


# ... (rest of the code remains the same for auto_dispatch_calls, run_agent_worker, run_api_server_thread, run_api_mode)

async def auto_dispatch_calls():
    """Automatically dispatch calls from a queue or configuration"""
    logger.info("=" * 80)
    logger.info("🤖 AUTO-DISPATCH MODE")
    logger.info("=" * 80)
    
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        raise ValueError("❌ LiveKit credentials not set in environment")
    
    lk_api = api.LiveKitAPI(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )
    
    phone_numbers_file = "phone_numbers.txt"
    
    if os.path.exists(phone_numbers_file):
        with open(phone_numbers_file, "r") as f:
            phone_numbers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"📋 Loaded {len(phone_numbers)} phone numbers from {phone_numbers_file}")
    else:
        phone_numbers = ["+916203879448"]
        logger.info(f"⚠️  No phone_numbers.txt found, using default: {phone_numbers[0]}")
    
    logger.info(f"📞 Starting auto-dispatch for {len(phone_numbers)} phone number(s)")
    
    for idx, phone_number in enumerate(phone_numbers, 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"📞 Dispatching call {idx}/{len(phone_numbers)}: {phone_number}")
            logger.info(f"{'='*80}")
            
            room_name = f"call-{phone_number.replace('+', '')}-{int(perf_counter() * 1000)}"
            
            await lk_api.room.create_room(
                api.CreateRoomRequest(name=room_name)
            )
            logger.info(f"✅ Created room: {room_name}")
            
            dispatch = await lk_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=room_name,
                    agent_name="outbound-caller",
                    metadata=phone_number,
                )
            )
            logger.info(f"✅ Dispatch created successfully for {phone_number}")
            
            # Wait between calls
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error dispatching call to {phone_number}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ All {len(phone_numbers)} calls dispatched")
    logger.info("="*80)
    
    await lk_api.aclose()


def run_agent_worker():
    """Run the LiveKit agent worker (must be in main thread for signal handling)"""
    import sys
    from datetime import datetime
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING LIVEKIT AGENT WORKER")
    logger.info("=" * 80)
    
    agent_worker_status["running"] = True
    agent_worker_status["started_at"] = datetime.now().isoformat()
    
    try:
        # Clear sys.argv to prevent LiveKit CLI from parsing our custom arguments
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], "dev"]
        
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="outbound-caller",
                prewarm_fnc=prewarm,
            )
        )
        
        sys.argv = original_argv
    except Exception as e:
        logger.error(f"❌ Agent worker error: {e}")
        agent_worker_status["running"] = False
        raise


def run_api_server_thread(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in a background thread"""
    logger.info("=" * 80)
    logger.info(f"🌐 STARTING API SERVER")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info("=" * 80)
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"❤️  Health Check: http://{host}:{port}/health")
    logger.info(f"📊 Status: http://{host}:{port}/status")
    logger.info(f"🌍 Language Stats: http://{host}:{port}/language-stats")
    logger.info(f"📞 Dispatch Call: POST http://{host}:{port}/dispatch/call")
    logger.info("=" * 80)
    
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
        raise ValueError("❌ SIP_OUTBOUND_TRUNK_ID is not set or invalid")
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 LIVEKIT OUTBOUND CALLER - DUAL-STT API MODE (FIXED)")
    logger.info("=" * 80)
    logger.info("Components:")
    logger.info("  ✅ Deepgram STT (Primary - High Accuracy Transcription)")
    logger.info("  ✅ Groq Whisper (Secondary - Language Detection)")
    logger.info("  ✅ FastAPI Server (Background Thread)")
    logger.info("  ✅ LiveKit Agent Worker (Main Thread)")
    logger.info("  ✅ Enhanced Logging (VAD, STT, Language Detection)")
    logger.info("=" * 80)
    logger.info("Supported Languages:")
    logger.info("  🇬🇧 English")
    logger.info("  🇮🇳 Hindi (हिंदी)")
    logger.info("  🇮🇳 Gujarati (ગુજરાતી)")
    logger.info("  🇮🇳 Kannada (ಕನ್ನಡ)")
    logger.info("=" * 80 + "\n")
    
    # Validate API keys
    if not deepgram_api_key:
        logger.error("❌ DEEPGRAM_API_KEY not set!")
        raise ValueError("DEEPGRAM_API_KEY is required")
    
    if not groq_api_key:
        logger.error("❌ GROQ_API_KEY not set!")
        raise ValueError("GROQ_API_KEY is required")
    
    logger.info("✅ API keys validated")
    
    # Get API configuration
    api_port = int(os.getenv("API_PORT", "8000"))
    api_host = os.getenv("API_HOST", "0.0.0.0")
    
    # Start API server in background thread
    logger.info("🚀 Starting API server in background thread...")
    api_thread = threading.Thread(
        target=run_api_server_thread,
        args=(api_host, api_port),
        daemon=True,
        name="APIServer"
    )
    api_thread.start()
    
    # Give API server time to start
    import time
    logger.info("⏳ Waiting for API server to initialize...")
    time.sleep(2)
    
    if api_thread.is_alive():
        logger.info("✅ API server thread is running!")
    else:
        logger.error("❌ API server thread failed to start!")
        raise RuntimeError("API server failed to start")
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Starting LiveKit Agent Worker in main thread...")
    logger.info("=" * 80 + "\n")
    
    # Start agent worker in main thread (needs signal handling)
    run_agent_worker()


if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError("❌ SIP_OUTBOUND_TRUNK_ID is not set or invalid")
    
    import sys
    
    # Print banner
    print("\n" + "=" * 80)
    print("🎙️  LIVEKIT MULTILINGUAL OUTBOUND CALLER - DUAL-STT (FIXED)")
    print("=" * 80)
    print("Version: 2.1.0")
    print("Features: Deepgram + Groq Whisper + Enhanced Logging")
    print("=" * 80 + "\n")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "api":
            logger.info("🔧 Mode: API (Production)")
            run_api_mode()
        
        elif mode == "auto":
            logger.info("🔧 Mode: Auto-Dispatch")
            asyncio.run(auto_dispatch_calls())
        
        elif mode == "dev":
            logger.info("🔧 Mode: Development (Agent Worker Only)")
            cli.run_app(
                WorkerOptions(
                    entrypoint_fnc=entrypoint,
                    agent_name="outbound-caller",
                    prewarm_fnc=prewarm,
                )
            )
        else:
            print("❌ Invalid mode specified!")
            print("\nUsage:")
            print("  python agent.py api   - Run API server with agent worker (production)")
            print("  python agent.py dev   - Run agent worker only (development)")
            print("  python agent.py auto  - Auto-dispatch calls from phone_numbers.txt")
            print("\nExamples:")
            print("  python agent.py api")
            print("  python agent.py dev")
            sys.exit(1)
    else:
        # Default to API mode
        logger.info("🔧 Mode: API (Default)")
        run_api_mode()