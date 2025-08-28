import os
import time
import uuid
import tempfile
import requests
from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time

# === Local dev: load .env if present (DO NOT commit .env) ===
load_dotenv()

# === CONFIG (from environment) ===
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus2")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

Api_Base_url = os.getenv("PUBLIC_BASE_URL")
# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Folder to store generated audio files (publicly served)
BASE_DIR = os.path.dirname(__file__)
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__)
call_audio_map = {}

# === Helper: time-based cleanup ===
def cleanup_old_audio_files(max_age_seconds=300):
    now = time.time()
    for filename in os.listdir(STATIC_FOLDER):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(STATIC_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        print(f"[CLEANUP] Deleted old file: {filename}")
                        app.logger.info(f"[CLEANUP] Deleted old file: {filename}")
                    except Exception as e:
                        print(f"[CLEANUP ERROR] Failed to delete {filename}: {e}")
                        app.logger.exception(f"[CLEANUP ERROR] Failed to delete {filename}")

# === Health check ===
@app.route("/", methods=["GET"])
def health_check():
    # print("[DEBUG] Health check hit")
    return Response("route url hello world", content_type="text/plain")

# === MAKE OUTBOUND CALL ===
@app.route("/makecall", methods=["POST"])
def make_call():
    print("\n[DEBUG] /makecall endpoint hit.")
    app.logger.info("[DEBUG] /makecall endpoint hit.")
    try:
        data = request.get_json(silent=True) or {}
        to_number = data.get("phone")
        print(f"[DEBUG] Request body: {data}")
        app.logger.info(f"[INFO] Making call to {to_number}")
        
        if not to_number:
            print("[ERROR] Phone number is required")
            app.logger.error("[ERROR] Phone number is required")
            return {"success": False, "message": "Phone number is required"}, 400

        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            print("[ERROR] Twilio credentials missing")
            app.logger.error("[ERROR] Twilio credentials missing")
            return {"success": False, "message": "Twilio credentials missing"}, 500

        client_twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        base_url =Api_Base_url
        if not base_url:
            print("[ERROR] PUBLIC_BASE_URL not set")
            app.logger.error("[ERROR] PUBLIC_BASE_URL not set")
            return {"success": False, "message": "Set PUBLIC_BASE_URL env var for callback URLs"}, 500

        call = client_twilio.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{base_url}/answer",
            machine_detection="Enable",
            status_callback=f"{base_url}/status",
            status_callback_event=["completed"]
        )

        print(f"[INFO] Outbound call initiated. SID: {call.sid}")
        app.logger.info(f"[INFO] Call SID: {call.sid}")
        return {"success": True, "sid": call.sid}

    except Exception as e:
        print(f"[ERROR] Failed to make call: {str(e)}")
        app.logger.exception("[ERROR] Failed to make call")
        return {"success": False, "message": str(e)}, 500

# === ANSWER CALL ===
@app.route("/answer", methods=["POST"])
def answer_call():
    try:
        print("answer_call endpoinnt hit")
        repeat = request.args.get("repeat", "false").lower() == "true"
        detected_lang = request.args.get("lang", "en-IN")
        
        print(f"[INFO] Answering call. Repeat: {repeat}")
        print(f"[INFO] Detected language: {detected_lang}")
        app.logger.info(f"[INFO] Answering call. Repeat: {repeat}")
        app.logger.info(f"[INFO] Detected language: {detected_lang}")
        resp = VoiceResponse()

        if repeat:
            print("inside repeat block, repeat is true")
            print(f"detected_lang: {detected_lang}")
            if(detected_lang == "hi-IN"):
                resp.say("आप फिर से सवाल पूछ सकते हैं।", voice="Polly.Anika", language="hi-IN")
            elif(detected_lang == "kn-IN"):
                resp.say("ಈಗ ನೀವು ಮತ್ತೊಂದು ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಬಹುದು.", voice="Polly.Anika", language="kn-IN")
            elif(detected_lang == "mr-IN"): 
                resp.say("तुम्ही पुन्हा प्रश्न विचारू शकता.", voice="Polly.Anika", language="mr-IN")
            else:   
                resp.say("You can ask another question now.", voice="Polly.Joanna", language="en-IN")
        else:
            resp.say("Hello, I am calling from LPU.", voice="Polly.Joanna", language="en-IN")
            resp.say("Please ask your question after the beep.", voice="Polly.Joanna", language="en-IN")
            # app.logger.info(f"[audio base url is]: {api}")

        #  added wav here in recording line 
        resp.record(
            action="/process_recording",  
            method="POST", max_length=30, 
            timeout=3, play_beep=True
            )
        
        return Response(str(resp), mimetype="text/xml")
    except Exception as e:
        print(f"[ERROR] /answer route failed: {e}")
        app.logger.exception("[ERROR] Failed in /answer")
        return Response("<Response><Say>Error occurred</Say></Response>", mimetype="text/xml")

# === PROCESS RECORDING ===
@app.route("/process_recording", methods=["POST"])
def process_recording():
    start_time = time.time()  # Total start time
    try:
        recording_url = request.form.get("RecordingUrl")
        call_sid = request.form.get("CallSid")
        print("recoring_url:", recording_url)
        print(f"[DEBUG] Recording URL: {recording_url}, CallSid: {call_sid}")
        app.logger.info(f"[DEBUG] Recording URL: {recording_url}, CallSid: {call_sid}")

        # --- Step 1: Download Twilio recording ---
        stt_start = time.time()
        audio_response = requests.get(
            recording_url + ".wav",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=20
        )
        print(f"audio url is ", recording_url)
        audio_response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_response.content)
            temp_audio_path = temp_audio.name
        print(f"[TIMING] Audio download time: {time.time() - stt_start:.2f} sec")

        # --- Step 2: STT ---
        stt_process_start = time.time()
        transcription, detected_lang = transcribe_with_azure(temp_audio_path)
        stt_process_end = time.time()
        print(f"[TRANSCRIPTION]: {transcription}")
        print(f"[DETECTED LANG]: {detected_lang}")
        app.logger.info(f"[TRANSCRIPTION]: {transcription}")
        app.logger.info(f"[DETECTED LANG]: {detected_lang}")
        print(f"[TIMING] STT processing time: {stt_process_end - stt_process_start:.2f} sec")

        if not transcription:
            resp = VoiceResponse()
            resp.say("Sorry, I didn't catch that. Please speak clearly after the beep.", voice="Polly.Joanna", language="en-IN")
            resp.redirect("/answer?repeat=true")
            return Response(str(resp), mimetype="text/xml")

        # --- Step 3: Build Prompt ---
        prompt_start = time.time()
        if detected_lang == "hi-IN":
            prompt_text = (
                f"The user asked (in Hindi): '{transcription}'. "
                "You are an official assistant for Lovely Professional University (LPU). "
                "Answer in Hinglish (mix Hindi + English), keep it polite and short. "
                "Provide only helpful details about LPU courses, admissions, fees, campus, and other official information. "
                "If you don't know, politely suggest contacting the LPU helpline."
            )
        elif detected_lang == "kn-IN":
            prompt_text = (
                f"The user asked (in Kannada): '{transcription}'. "
                "You are an official assistant for Lovely Professional University (LPU). "
                "Answer in Kannada, keep it polite and short. "
                "Provide only helpful details about LPU courses, admissions, fees, campus, and other official information. "
                "If you don't know, politely suggest contacting the LPU helpline."
            )
        elif detected_lang == "mr-IN":
            prompt_text = (
                f"The user asked (in Marathi): '{transcription}'. "
                "You are an official assistant for Lovely Professional University (LPU). "
                "Answer in Marathi, keep it polite and short. "
                "Provide only helpful details about LPU courses, admissions, fees, campus, and other official information. "
                "If you don't know, politely suggest contacting the LPU helpline."
            )
        else:
            prompt_text = (
                f"The user asked (in English): '{transcription}'. "
                "You are an official assistant for Lovely Professional University (LPU). "
                "Answer in English, keep it polite and short. "
                "Provide only helpful details about LPU courses, admissions, fees, campus, and other official information. "
                "If you don't know, politely suggest contacting the LPU helpline."
            )
        prompt_end = time.time()
        print(f"[TIMING] Prompt building time: {prompt_end - prompt_start:.2f} sec")

        # --- Step 4: AI Response ---
        ai_start = time.time()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}]
        )
        ai_response = completion.choices[0].message.content.strip()
        ai_end = time.time()
        print(f"[AI RESPONSE]: {ai_response}")
        app.logger.info(f"[AI RESPONSE]: {ai_response}")
        print(f"[TIMING] AI generation time: {ai_end - ai_start:.2f} sec")

        # --- Step 5: TTS ---
        tts_start = time.time()
        audio_url, filename = generate_elevenlabs_audio(ai_response, call_sid)
        tts_end = time.time()
        print(f"[AUDIO URL]: {audio_url}")
        app.logger.info(f"[AUDIO URL]: {audio_url}")
        print(f"[TIMING] TTS generation time: {tts_end - tts_start:.2f} sec")

        # --- Total ---
        total_time = time.time() - start_time
        print(f"[TIMING] Total /process_recording time: {total_time:.2f} sec")

        # Play audio
        resp = VoiceResponse()
        resp.play(audio_url)
        public_base_url = Api_Base_url
        resp.redirect(f"{public_base_url}/answer?repeat=true&lang={detected_lang}")
        return Response(str(resp), mimetype="text/xml")

    except Exception as e:
        print(f"[ERROR] Processing recording failed: {e}")
        app.logger.exception("[ERROR] Failed in /process_recording")
        resp = VoiceResponse()
        resp.say("Sorry, an error occurred while processing your request.", voice="Polly.Joanna", language="en-IN")
        return Response(str(resp), mimetype="text/xml")


# === AZURE STT ===
def transcribe_with_azure(audio_path):
    try:
        print(f"[DEBUG] Transcribing with Azure: {audio_path}")
        app.logger.info(f"[DEBUG] Transcribing with Azure: {audio_path}")
        
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        auto_detect_config = speechsdk.AutoDetectSourceLanguageConfig(languages=["en-IN", "hi-IN","kn-IN","mr-IN"])
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config, 
            auto_detect_source_language_config=auto_detect_config
            )
        
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
            return result.text.strip(), detected
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "", ""
        else:
            raise Exception(f"STT failed: {result.reason}")
    except Exception as e:
        print(f"[STT ERROR]: {e}")
        app.logger.exception("[STT ERROR]")
        return "", ""

# === ELEVEN LABS TTS ===
def generate_elevenlabs_audio(text, call_sid=None):
    try:
        # cleanup_old_audio_files(max_age_seconds=300)
        print(f"[DEBUG] Generating ElevenLabs TTS for text: {text}")
        app.logger.info(f"[DEBUG] Generating ElevenLabs TTS for text: {text}")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
        headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()

        filename = f"{uuid.uuid4().hex}.mp3"
        full_path = os.path.join(STATIC_FOLDER, filename)
        with open(full_path, "wb") as f:
            f.write(r.content)

        if call_sid:
            call_audio_map[call_sid] = filename

        public_base = os.getenv("PUBLIC_BASE_URL", request_base_url())
        return f"{public_base.rstrip('/')}/static/{filename}", filename

    except Exception as e:
        print(f"[TTS ERROR]: {e}")
        app.logger.exception("[TTS ERROR]")
        raise

def request_base_url():
    try:
        return request.url_root
    except RuntimeError:
        return os.getenv("PUBLIC_BASE_URL", "")

# === STATIC FILES ===
@app.route("/static/<filename>")
def serve_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=False)

# === STATUS CALLBACK ===
@app.route("/status", methods=["POST"])
def call_status():
    try:
        call_sid = request.form.get("CallSid")
        call_status = request.form.get("CallStatus")
        print(f"[STATUS] CallSid={call_sid} Status={call_status}")
        app.logger.info(f"[STATUS] CallSid={call_sid} Status={call_status}")
        
        if call_sid and call_status in ("completed", "failed", "busy", "no-answer"):
            filename = call_audio_map.pop(call_sid, None)
            if filename:
                file_path = os.path.join(STATIC_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[STATUS CLEANUP] Deleted audio: {filename}")
                    app.logger.info(f"[STATUS CLEANUP] Deleted audio: {filename}")
                    
        return ("", 204)
    except Exception as e:
        print(f"[STATUS ERROR]: {e}")
        return ("", 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=(os.getenv("FLASK_DEBUG", "false").lower()=="true"))
