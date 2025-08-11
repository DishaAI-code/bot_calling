import os
import time
import uuid
import tempfile
import requests
from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import openai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# === Local dev: load .env if present (DO NOT commit .env) ===
load_dotenv()

# === CONFIG (from environment) ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # set in Azure App Settings
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus2")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # when making outbound calls

# Folder to store generated audio files (publicly served)
BASE_DIR = os.path.dirname(__file__)
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__)

# Map CallSid -> generated filename so we can cleanup after call ends
call_audio_map = {}

# === Helper: time-based cleanup (fallback) ===
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
                        app.logger.info(f"[CLEANUP] Deleted old file: {filename}")
                    except Exception as e:
                        app.logger.exception(f"[CLEANUP ERROR] Failed to delete {filename}: {e}")

# === Health check ===
@app.route("/", methods=["GET"])
def health_check():
    return Response("LPU Course Bot is running", content_type="text/plain")

# === Make outbound call (optional) ===
@app.route("/make-call", methods=["POST"])
def make_call():
    try:
        data = request.get_json() or {}
        to_number = data.get("phone")
        if not to_number:
            return {"success": False, "message": "Phone number is required"}, 400

        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            return {"success": False, "message": "Twilio credentials missing"}, 500

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # ensure URL points to your deployed /answer and /status endpoints
        base_url = os.getenv("PUBLIC_BASE_URL")  # e.g. https://your-app.azurewebsites.net
        if not base_url:
            return {"success": False, "message": "Set PUBLIC_BASE_URL env var for callback URLs"}, 500

        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{base_url}/answer",
            machine_detection="Enable",
            status_callback=f"{base_url}/status",
            status_callback_event=["completed"]  # you can also include initiated, ringing, answered
        )

        return {"success": True, "sid": call.sid}
    except Exception as e:
        print(f"[ERROR] Failed to log request: {e}")
        return {"success": False, "message": "Internal server error"}, 500
    
# === 1. ANSWER CALL ===
@app.route("/answer", methods=["POST"])
def answer_call():
    repeat = request.args.get("repeat", "false").lower() == "true"
    app.logger.info(f"[INFO] Answering call, repeat={repeat}")
    resp = VoiceResponse()
    if repeat:
        resp.say("You can ask another question now.", voice="Polly.Joanna", language="en-IN")
    else:
        resp.say("Hello, I am calling from LPU.", voice="Polly.Joanna", language="en-IN")
        resp.say("Please ask your question after the beep.", voice="Polly.Joanna", language="en-IN")

    resp.record(
        action="/process_recording",
        method="POST",
        max_length=30,
        timeout=3,
        play_beep=True
    )
    return Response(str(resp), mimetype="text/xml")

# === PROCESS RECORDING ===
@app.route("/process_recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordingUrl")
    call_sid = request.form.get("CallSid")
    app.logger.info(f"[INFO] Recording URL: {recording_url}, CallSid: {call_sid}")

    try:
        # download Twilio recording as wav
        audio_response = requests.get(recording_url + ".wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
        audio_response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_response.content)
            temp_audio_path = temp_audio.name

        # transcribe with Azure (auto-detect en/hi)
        transcription, detected_lang = transcribe_with_azure(temp_audio_path)
        app.logger.info("[TRANSCRIPTION]: %s", transcription)
        app.logger.info("[DETECTED LANG]: %s", detected_lang)

        if not transcription:
            resp = VoiceResponse()
            resp.say("Sorry, I didn't catch that. Please speak clearly after the beep.", voice="Polly.Joanna", language="en-IN")
            resp.redirect("/answer?repeat=true")
            return Response(str(resp), mimetype="text/xml")

        if detected_lang and detected_lang.startswith("hi"):
            prompt_text = f"""
            The user said: '{transcription}'.
            Respond in Hinglish (Hindi using English letters).
            Keep your answer short, helpful, and clear.
            The topic is always about LPU admissions and courses.
            """
        else:
            prompt_text = f"""
            The user said: '{transcription}'.
            Respond in English.
            Keep your answer short, helpful, and clear.
            The topic is always about LPU admissions and courses.
            """

        # Get response from OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}]
        )
        ai_response = completion["choices"][0]["message"]["content"].strip()
        app.logger.info("[AI RESPONSE]: %s", ai_response)

        # Generate audio (ElevenLabs) and save to static folder
        audio_url, filename = generate_elevenlabs_audio(ai_response, call_sid)
        app.logger.info("[AUDIO URL]: %s", audio_url)

        # Build TwiML to play audio and then redirect to /answer for repeat
        resp = VoiceResponse()
        # request.url_root already contains trailing slash in many setups; rstrip to be safe
        resp.play(audio_url)
        resp.redirect("/answer?repeat=true")
        return Response(str(resp), mimetype="text/xml")

    except Exception as e:
        app.logger.exception("[ERROR] processing recording")
        resp = VoiceResponse()
        resp.say("Sorry, an error occurred while processing your request.", voice="Polly.Joanna", language="en-IN")
        return Response(str(resp), mimetype="text/xml")

# === AZURE STT ===
def transcribe_with_azure(audio_path):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        auto_detect_config = speechsdk.AutoDetectSourceLanguageConfig(languages=["en-IN", "hi-IN"])
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config
        )
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = result.properties.get(speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult)
            return result.text.strip(), detected
        elif result.reason == speechsdk.ResultReason.NoMatch:
            app.logger.info("[STT] No speech recognized")
            return "", ""
        else:
            raise Exception(f"STT failed: {result.reason}")
    except Exception as e:
        app.logger.exception("[STT ERROR]")
        return "", ""

# === ELEVEN LABS TTS AND SAVE FILE ===
def generate_elevenlabs_audio(text, call_sid=None):
    """
    Generates TTS via ElevenLabs, saves to STATIC_FOLDER, and records the mapping
    so we can delete the file when the call completes.
    Returns (public_url, filename)
    """
    # fallback cleanup (time-based) to prevent disk bloat
    cleanup_old_audio_files(max_age_seconds=300)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()

    filename = f"{uuid.uuid4().hex}.mp3"
    full_path = os.path.join(STATIC_FOLDER, filename)
    with open(full_path, "wb") as f:
        f.write(r.content)

    # if CallSid provided, store it for deletion on /status
    if call_sid:
        call_audio_map[call_sid] = filename
    else:
        # if there's no call SID (e.g., manual test), store with a generated id and let fallback cleanup handle it
        app.logger.info("[TTS] No CallSid provided - relying on time-based cleanup")

    # PUBLIC URL used by Twilio. Make sure PUBLIC_BASE_URL env var is set to your deployed app URL.
    public_base = os.getenv("PUBLIC_BASE_URL", request_base_url())
    public_url = f"{public_base.rstrip('/')}/static/{filename}"
    return public_url, filename

def request_base_url():
    # As a best-effort fallback when request context is present
    try:
        return request.url_root
    except RuntimeError:
        return os.getenv("PUBLIC_BASE_URL", "")

# === Serve static audio files ===
@app.route("/static/<filename>")
def serve_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=False)

# === Status callback from Twilio to cleanup after call ends ===
@app.route("/status", methods=["POST"])
def call_status():
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")
    app.logger.info(f"[STATUS] CallSid={call_sid} status={call_status}")

    # Only cleanup when call completed (or you can choose 'completed'/'failed'/'busy' etc.)
    if call_sid and call_status in ("completed", "failed", "busy", "no-answer"):
        filename = call_audio_map.pop(call_sid, None)
        if filename:
            file_path = os.path.join(STATIC_FOLDER, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    app.logger.info(f"[STATUS CLEANUP] Deleted audio for {call_sid}: {filename}")
                except Exception as e:
                    app.logger.exception(f"[STATUS CLEANUP ERROR] Could not delete {file_path}: {e}")
    return ("", 204)

# === Run (listen on 0.0.0.0 for Azure) ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=(os.getenv("FLASK_DEBUG", "false").lower()=="true"))
