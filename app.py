# this is actually a agent.py - renamining for testing purpose only

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

from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import openai, silero
from groq import Groq
from livekit.plugins import sarvam
from livekit.agents import metrics, MetricsCollectedEvent, ConversationItemAddedEvent
from livekit.agents.llm import ImageContent, AudioContent
from livekit.agents import UserInputTranscribedEvent

# Langfuse SDK for better tracing
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

# Load environment variables
load_dotenv(dotenv_path=".env")

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.DEBUG)

stt_logger = logging.getLogger("STT")
stt_logger.setLevel(logging.DEBUG)

lang_detect_logger = logging.getLogger("LanguageDetection")
lang_detect_logger.setLevel(logging.DEBUG)

vad_logger = logging.getLogger("VAD")
vad_logger.setLevel(logging.DEBUG)

# ============ ENVIRONMENT VARIABLES ============
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
groq_api_key = os.getenv("GROQ_API_KEY")
sarvam_api_key = os.getenv("sarvam_api_key")

# Validate critical API keys
if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
    logger.error("SIP_OUTBOUND_TRUNK_ID not set or invalid!")
    
if not groq_api_key:
    logger.error(" GROQ_API_KEY not set in environment!")

if not sarvam_api_key:
    logger.error("sarvam_api_key not set in environment!")

# Global Langfuse client
langfuse_client = None

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
- Kannada (ಕನ್ನડ)

CONVERSATION STYLE:
4. Keep responses short (1-3 sentences), friendly, and conversational.
5. Do not repeat the welcome greeting - it has already been said.
6. You are calling from Autodesk to assist the user with product information and queries.
7. Allow the user to end the conversation naturally.
8. Use natural, colloquial expressions appropriate for phone conversations in each language.

IMPORTANT: Respond in the user's detected language. The system has already identified it for you."""

_greeting_message = "Hello, I am your AI assistant calling from Autodesk. I can understand and speak English, Hindi, Gujarati, and Kannada. How can I help you today?"


# ============ LANGUAGE DETECTION WITH WHISPER (GROQ) ============

class LanguageDetector:
    """Handles language detection using Whisper on Groq"""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.current_language = "en"
        self.detection_count = 0
        lang_detect_logger.info("✅ LanguageDetector initialized with Groq Whisper")
    
    async def detect_language_from_frames(self, audio_frames: list) -> str | None:
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
            lang_detect_logger.info(f"✅ Whisper API response time: {elapsed:.3f}s")
            
            # Extract language from response
            detected_language = getattr(response, 'language', None)
            transcript = getattr(response, 'text', '')
            
            if detected_language:
                lang_code = self._normalize_language_code(detected_language)
                
                lang_detect_logger.info("=" * 80)
                lang_detect_logger.info(f"✅ LANGUAGE DETECTED: {detected_language} ({lang_code})")
                lang_detect_logger.info(f"📝 Transcript: {transcript}")
                lang_detect_logger.info(f"⏱️  Detection time: {elapsed:.3f}s")
                lang_detect_logger.info("=" * 80)
                
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


# ============ CUSTOM AGENT WITH LANGUAGE DETECTION ============

class MultilingualAgent(Agent):
    """Extended Agent with multilingual language detection support"""
    
    def __init__(self, language_detector: LanguageDetector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_detector = language_detector
        self.current_stt_language = "en-US"
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
            lang_detect_logger.info("🔄 Starting language detection process...")
            
            # Detect language using Whisper
            detected_lang = await self.language_detector.detect_language_from_frames(audio_frames)
            
            if detected_lang:
                current_lang = self.current_stt_language.split('-')[0]
                
                if detected_lang != current_lang:
                    lang_detect_logger.info("=" * 80)
                    lang_detect_logger.info(f"🔀 LANGUAGE SWITCH DETECTED!")
                    lang_detect_logger.info(f"   FROM: {self.current_stt_language}")
                    lang_detect_logger.info(f"   TO: {detected_lang}")
                    lang_detect_logger.info("=" * 80)
                    
                    # Map to Sarvam language codes
                    sarvam_lang_map = {
                        'en': 'en-IN',
                        'hi': 'hi-IN',
                        'gu': 'gu-IN',
                        'kn': 'kn-IN',
                        'ta': 'ta-IN',
                        'te': 'te-IN',
                        'ml': 'ml-IN',
                        'bn': 'bn-IN',
                        'mr': 'mr-IN',
                    }
                    
                    new_sarvam_lang = sarvam_lang_map.get(detected_lang, 'en-IN')
                    lang_detect_logger.info(f"📍 Language mapping: {detected_lang} → {new_sarvam_lang}")
                    
                    # Update STT language
                    self.current_stt_language = new_sarvam_lang
                    
                    if hasattr(self, '_stt_instance'):
                        stt_logger.info("=" * 80)
                        stt_logger.info(f"✅ STT UPDATED")
                        stt_logger.info(f"   New language: {new_sarvam_lang}")
                        stt_logger.info("=" * 80)
                
        except Exception as e:
            lang_detect_logger.error(f"❌ Error in language detection: {e}")
            import traceback
            traceback.print_exc()


# ============ LLM INTERACTION LOGGING ============

async def log_llm_interaction(user_input: str, llm_response: str, session_metadata: dict = None):
    """Log LLM input/output interactions"""
    print("=" * 80)
    print("💬 LLM INTERACTION LOG")
    print("=" * 80)
    print(f"🎤 USER INPUT:\n{user_input}")
    print("-" * 80)
    print(f"🤖 AGENT RESPONSE:\n{llm_response}")
    print("=" * 80)
    
    logger.info(f"LLM Interaction logged - User: {user_input[:50]}... → Agent: {llm_response[:50]}...")


# ============ MAIN AGENT CODE ============

async def entrypoint(ctx: JobContext):
    """Main entry point for LiveKit agent worker"""
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
    logger.info(f"📱 Dialing {phone_number} to room {ctx.room.name}")

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
    logger.info("📊 Monitoring call status...")
    
    while perf_counter() - start_time < 300:  # 5 minute timeout
        call_status = participant.attributes.get("sip.callStatus")
        
        if call_status == "active":
            logger.info("✅ User has picked up - call is active")
            return
        elif call_status == "automation":
            logger.debug("⏳ Call status: automation")
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
            logger.info("❌ User rejected the call, exiting job")
            break
        elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
            logger.info("❌ User did not pick up, exiting job")
            break
            
        await asyncio.sleep(0.1)

    logger.info("⏱️  Session ended")
    ctx.shutdown()


async def run_voice_agent(
    ctx: JobContext, 
    participant: rtc.RemoteParticipant, 
    instructions: str,
    greeting_message: str
):
    """Run the voice agent for conversation"""
    logger.info("=" * 80)
    logger.info("🎤 INITIALIZING VOICE AGENT")
    logger.info("=" * 80)
    
    # Validate API keys
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY is not set in environment")
    
    logger.info("✅ API Keys validated")
    
    # Initialize Language Detector
    logger.info("🔍 Initializing Language Detector (Groq Whisper)...")
    language_detector = LanguageDetector(groq_api_key)
    
    # Initialize Sarvam STT
    logger.info("🎙️  Initializing Sarvam STT...")
    stt_instance = sarvam.STT(
        language="unknown",
        model="saarika:v2.5"
    )
    logger.info("✅ Sarvam STT initialized")

    # Initialize LLM
    logger.info("🤖 Initializing GPT-4 LLM...")
    llm_instance = openai.LLM(
        model="gpt-4",
        temperature=0.7,
    )
    logger.info("✅ GPT-4 LLM initialized")

    # Initialize TTS
    logger.info("🔊 Initializing OpenAI TTS...")
    tts_instance = openai.TTS(
        voice="alloy",
        speed=1.0,
    )
    logger.info("✅ OpenAI TTS initialized")

    # Create multilingual agent
    logger.info("🤖 Creating MultilingualAgent...")
    assistant = MultilingualAgent(
        language_detector=language_detector,
        instructions=instructions,
    )
    assistant._stt_instance = stt_instance

    # Create agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
    )

    # ============ EVENT HANDLERS ============

    @session.on("user_speech_committed")
    def on_user_speech_committed(event):
        """Handle user speech"""
        try:
            if hasattr(event, 'alternatives') and event.alternatives and len(event.alternatives) > 0:
                transcript = event.alternatives[0].text
                stt_logger.info(f"👤 USER: {transcript}")
                        
            elif hasattr(event, 'text'):
                transcript = event.text
                stt_logger.info(f"👤 USER: {transcript}")
                
        except Exception as e:
            stt_logger.error(f"❌ Error in user_speech_committed: {e}")

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
        """Handle agent speech responses"""
        try:
            if hasattr(event, 'text'):
                agent_text = event.text
                logger.info(f"🤖 AGENT: {agent_text}")
                        
            elif hasattr(event, 'alternatives') and event.alternatives and len(event.alternatives) > 0:
                agent_text = event.alternatives[0].text
                logger.info(f"🤖 AGENT: {agent_text}")
                
        except Exception as e:
            logger.error(f"❌ Error in agent_speech_committed: {e}")

    @session.on("any")
    def on_any_event(event_name, *args):
        """Debug events"""
        if event_name not in ["audio_stream"]:
            logger.debug(f"📡 EVENT: {event_name}")

    # Metrics collection
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        logger.info("📊 Metrics collected")
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📈 Usage Summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Conversation logging
    conversation_buffer = {
        "last_user_input": None,
        "last_agent_response": None,
    }

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        """Log conversation items"""
        try:
            role = event.item.role
            text_content = event.item.text_content
            
            for content in event.item.content:
                if isinstance(content, str):
                    if role == "user":
                        logger.info(f"👤 USER INPUT: {content}")
                        conversation_buffer["last_user_input"] = content
                    elif role == "assistant":
                        logger.info(f"🤖 AGENT OUTPUT: {content}")
                        conversation_buffer["last_agent_response"] = content
                        
                        if conversation_buffer["last_user_input"]:
                            asyncio.create_task(
                                log_llm_interaction(
                                    user_input=conversation_buffer["last_user_input"],
                                    llm_response=content,
                                    session_metadata={
                                        "room": ctx.room.name,
                                        "language": assistant.current_stt_language,
                                    }
                                )
                            )
                            conversation_buffer["last_user_input"] = None
                            conversation_buffer["last_agent_response"] = None
                
        except Exception as e:
            logger.error(f"❌ Error in conversation_item_added: {e}")

    try:
        # Start the session
        await session.start(room=ctx.room, agent=assistant)
        logger.info("✅ Agent session started successfully")
        
        await asyncio.sleep(1.5)
        
        # Send greeting message
        logger.info(f"📢 Sending greeting: {greeting_message}")
        await session.say(greeting_message, allow_interruptions=True)
        logger.info("✅ Greeting sent")
        
    except Exception as e:
        logger.error(f"❌ Error in voice agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Flush tracing
        if langfuse_client and LANGFUSE_AVAILABLE:
            try:
                langfuse_client.flush()
                logger.info("📤 Langfuse events flushed")
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")


def prewarm(proc: JobProcess):
    """Prewarm VAD model"""
    logger.info("=" * 80)
    logger.info("🔥 PREWARMING VAD MODEL")
    logger.info("=" * 80)
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("✅ VAD model prewarmed successfully")


# ============ MAIN ENTRY POINT ============

if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError("❌ SIP_OUTBOUND_TRUNK_ID is not set or invalid")
    
    print("\n" + "=" * 80)
    print("  🎯 LIVEKIT OUTBOUND CALLER - AGENT WORKER")
    print("=" * 80)
    print("Version: 2.1.0")
    print("Role: Agent Worker (Handles voice conversations)")
    print("=" * 80 + "\n")
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING LIVEKIT AGENT WORKER")
    logger.info("=" * 80)
    
    try:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="outbound-caller",
                prewarm_fnc=prewarm,
            )
        )
    except Exception as e:
        logger.error(f"❌ Agent worker error: {e}")
        import traceback
        traceback.print_exc()
        raise