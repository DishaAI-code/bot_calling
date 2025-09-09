# app2_fixed.py
import os
import time
import uuid
import tempfile
import requests
import threading
import traceback
from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# === Local dev: load .env if present (DO NOT commit .env) ===
load_dotenv()

# === CONFIG (from environment) ===
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "eastus2")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # keep your public base here

# Initialize OpenAI client (you already had this; not used heavily but kept)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Folder to store generated audio files (publicly served)
BASE_DIR = os.path.dirname(__file__)
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__)
call_audio_map = {}     # maps call_sid -> filename (when audio ready)
call_info_map = {}      # maps call_sid -> metadata (to number, created timestamp)

# === Query deployed LLM (unchanged) ===
def query_deployed_llm(prompt_text):
    try:
        app.logger.info(f"[DEBUG] Querying deployed LLM with prompt: {prompt_text}")
        url = "http://98.70.101.220:11434/api/generate"
        payload = {"model": "lpu-assistant", "prompt": prompt_text, "stream": False}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        resp_text = r.json().get("response", "").strip()
        print(f"LLM response: {resp_text}")
        app.logger.info(f"[LLM RESPONSE (len={len(resp_text)} chars)]")
        return resp_text
    except Exception as e:
        app.logger.exception("[LLM ERROR]")
        return "Sorry, I am unable to process your request at the moment."

# === Helper: cleanup old audio files (unchanged) ===
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
                    except Exception:
                        app.logger.exception(f"[CLEANUP ERROR] Failed to delete {filename}")

# === Health check ===
@app.route("/", methods=["GET"])
def health_check():
    return Response("route url hello world", content_type="text/plain")

# === MAKE OUTBOUND CALL ===
@app.route("/makecall", methods=["POST"])
def make_call():
    try:
        data = request.get_json(silent=True) or {}
        to_number = data.get("phone")
        app.logger.info(f"[makecall] payload={data}")

        if not to_number:
            return {"success": False, "message": "Phone number is required"}, 400

        client_twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client_twilio.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{Api_Base_url}/answer",
            machine_detection="Enable",
            status_callback=f"{Api_Base_url}/status",
            status_callback_event=["completed"]
        )

        # store call metadata so fallback can reach the user if needed
        call_info_map[call.sid] = {"to": to_number, "created": time.time()}
        app.logger.info(f"[makecall] created call sid={call.sid} to={to_number}")
        return {"success": True, "sid": call.sid}
    except Exception as e:
        app.logger.exception("[makecall ERROR]")
        return {"success": False, "message": str(e)}, 500

# === ANSWER CALL (unchanged behavior) ===
@app.route("/answer", methods=["POST"])
def answer_call():
    try:
        app.logger.info(f"[answer] request.form={dict(request.form)} args={dict(request.args)}")
        resp = VoiceResponse()
        repeat = request.args.get("repeat", "false").lower() == "true"
        detected_lang = request.args.get("lang", "en-IN")

        if repeat:
            if detected_lang == "hi-IN":
                resp.say("आप फिर से सवाल पूछ सकते हैं।", voice="Polly.Aditi", language="hi-IN")
            elif detected_lang == "kn-IN":
                resp.say("ಈಗ ನೀವು ಮತ್ತೊಂದು ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಬಹುದು.", voice="Polly.Aditi", language="kn-IN")
            elif detected_lang == "mr-IN":
                resp.say("तुम्ही पुन्हा प्रश्न विचारू शकता.", voice="Polly.Aditi", language="mr-IN")
            else:
                resp.say("You can ask another question now.", voice="Polly.Joanna", language="en-IN")
        else:
            resp.say("Hello, I am calling from LPU.", voice="Polly.Joanna", language="en-IN")
            resp.say("Please ask your question after the beep.", voice="Polly.Joanna", language="en-IN")

        # record; action points to /process_recording (unchanged)
        resp.record(action="/process_recording", method="POST", max_length=30, timeout=3, play_beep=True)
        return Response(str(resp), mimetype="text/xml")
    except Exception:
        app.logger.exception("[answer ERROR]")
        return Response("<Response><Say>Error occurred</Say></Response>", mimetype="text/xml")

# === WAIT endpoint: keeps the call alive until audio ready ===
@app.route("/wait", methods=["POST"])
def wait_for_audio():
    """
    Twilio will POST here repeatedly while we 'hold' the caller.
    If audio is ready (call_audio_map has filename), play it and exit.
    Otherwise, say a polite hold message, pause, and redirect back to /wait.
    """
    try:
        # callSid may be passed as query param or form param (Twilio)
        call_sid = request.args.get("call_sid") or request.form.get("CallSid")
        app.logger.info(f"[wait] hit for call_sid={call_sid}; call_audio_map_has={bool(call_audio_map.get(call_sid))}")

        filename = call_audio_map.get(call_sid)
        if filename:
            audio_url = f"{Api_Base_url}/static/{filename}"
            app.logger.info(f"[wait] audio ready for {call_sid}: {audio_url}")
            resp = VoiceResponse()
            resp.play(audio_url)
            # after playing, redirect back to /answer to allow repeat
            resp.redirect(f"{Api_Base_url}/answer?repeat=true")
            return Response(str(resp), mimetype="text/xml")
        else:
            # Not ready yet: polite hold music / message, then redirect back
            resp = VoiceResponse()
            resp.say("Please hold while I prepare your answer.", voice="Polly.Joanna", language="en-IN")
            # pause length can be tuned; smaller values cause more HTTP requests
            resp.pause(length=8)
            resp.redirect(f"{Api_Base_url}/wait?call_sid={call_sid}", method="POST")
            return Response(str(resp), mimetype="text/xml")

    except Exception:
        app.logger.exception("[wait ERROR]")
        resp = VoiceResponse()
        resp.say("Sorry, an error occurred.", voice="Polly.Joanna", language="en-IN")
        return Response(str(resp), mimetype="text/xml")

# === PLAY READY AUDIO endpoint (used for outbound fallback calls) ===
@app.route("/play_ready_audio", methods=["POST", "GET"])
def play_ready_audio():
    try:
        filename = request.args.get("file")
        detected_lang = request.args.get("lang", "en-IN")
        if not filename:
            return Response("<Response><Say>Audio not available</Say></Response>", mimetype="text/xml")
        audio_url = f"{Api_Base_url}/static/{filename}"
        resp = VoiceResponse()
        resp.play(audio_url)
        resp.redirect(f"{Api_Base_url}/answer?repeat=true&lang={detected_lang}")
        return Response(str(resp), mimetype="text/xml")
    except Exception:
        app.logger.exception("[play_ready_audio ERROR]")
        return Response("<Response><Say>Error occurred</Say></Response>", mimetype="text/xml")

# === ASYNC worker: STT -> LLM -> TTS -> try to play on live call ===
def async_process_recording(recording_url, call_sid):
    """
    This runs in a background thread. It:
      - downloads the Twilio recording
      - transcribes with Azure
      - builds the prompt (language aware)
      - queries your deployed LLM
      - TTS with ElevenLabs, saves file
      - stores filename in call_audio_map[call_sid]
      - tries to update the live call to play the file; if call ended, fallback to SMS + outbound call
    """
    started = time.time()
    client_twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        app.logger.info(f"[ASYNC] start for call_sid={call_sid}, recording_url={recording_url}")

        # --- Download recording ---
        dl_start = time.time()
        r = requests.get(recording_url + ".wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(r.content)
            tmp_audio_path = tmp.name
        app.logger.info(f"[ASYNC] downloaded recording in {time.time() - dl_start:.2f}s -> {tmp_audio_path}")

        # --- STT (Azure) ---
        stt_start = time.time()
        transcription, detected_lang = transcribe_with_azure(tmp_audio_path)
        app.logger.info(f"[ASYNC] transcription='{transcription}' detected_lang={detected_lang} (stt_time={time.time() - stt_start:.2f}s)")

        if not transcription:
            ai_response = "Sorry, I couldn't understand you. Please try again."
        else:
            # --- Build prompt (preserve your original prompt templates) ---
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

            # --- Query LLM ---
            ai_start = time.time()
            ai_response = query_deployed_llm(prompt_text)
            app.logger.info(f"[ASYNC] ai_response length={len(ai_response)} (ai_time={time.time() - ai_start:.2f}s)")

        # --- TTS (ElevenLabs) ---
        tts_start = time.time()
        audio_url, filename = generate_elevenlabs_audio(ai_response, call_sid)
        app.logger.info(f"[ASYNC] TTS saved -> {filename} (tts_time={time.time() - tts_start:.2f}s) audio_url={audio_url}")

        # --- Put filename into call_audio_map so /wait can pick it up ---
        call_audio_map[call_sid] = filename

        # --- Try to update the live call so it plays immediately ---
        try:
            resp = VoiceResponse()
            resp.play(audio_url)
            # after playing, send them back to /answer to allow followup
            resp.redirect(f"{Api_Base_url}/answer?repeat=true&lang={detected_lang or 'en-IN'}")

            app.logger.info(f"[ASYNC] attempting client.calls({call_sid}).update(twiml=...)")
            client_twilio.calls(call_sid).update(twiml=str(resp))
            app.logger.info(f"[ASYNC] successfully updated live call {call_sid} to play audio.")
        except TwilioRestException as e:
            # Twilio-specific handling: if call not in-progress (21220), we fallback
            app.logger.exception(f"[ASYNC] TwilioRestException while updating call {call_sid}: {e}")
            status_code = getattr(e, "code", None)
            if status_code == 21220:
                app.logger.warning(f"[ASYNC] call {call_sid} not in-progress. Running fallback (SMS + outbound call).")
                # fallback actions: 1) SMS link 2) start new outbound call to play audio
                info = call_info_map.get(call_sid, {})
                to_number = info.get("to")
                # SMS with link
                if to_number:
                    try:
                        client_twilio.messages.create(
                            body=f"Your LPU assistant answer is ready: {audio_url}",
                            from_=TWILIO_PHONE_NUMBER,
                            to=to_number
                        )
                        app.logger.info(f"[ASYNC] fallback SMS sent to {to_number}")
                    except Exception:
                        app.logger.exception("[ASYNC] failed to send fallback SMS")

                    # Place a new outbound call to play the audio
                    try:
                        client_twilio.calls.create(
                            to=to_number,
                            from_=TWILIO_PHONE_NUMBER,
                            url=f"{Api_Base_url}/play_ready_audio?file={filename}&lang={detected_lang or 'en-IN'}",
                            method="POST"
                        )
                        app.logger.info(f"[ASYNC] fallback outbound call initiated to {to_number} to play audio")
                    except Exception:
                        app.logger.exception("[ASYNC] failed to create fallback outbound call")
                else:
                    app.logger.warning(f"[ASYNC] no to_number found for call_sid {call_sid} - cannot do fallback call/SMS")
            else:
                app.logger.exception("[ASYNC] Twilio update failed for unknown reason")
        except Exception:
            app.logger.exception("[ASYNC] Unexpected error while updating call")

        app.logger.info(f"[ASYNC] finished for call_sid={call_sid} total_time={(time.time() - started):.2f}s")
    except Exception:
        app.logger.exception("[async_process_recording ERROR]")
        # ensure we don't leave callers waiting forever; map an error message file or set a flag if needed
        try:
            # create short TTS to notify user about error (optional; keep simple)
            err_text = "Sorry, an error occurred while preparing the response. Please try again later."
            audio_url, filename = generate_elevenlabs_audio(err_text, call_sid)
            call_audio_map[call_sid] = filename
        except Exception:
            app.logger.exception("[async_process_recording ERROR] failed to create fallback TTS")

# === PROCESS RECORDING (now returns immediately and redirects to /wait) ===
@app.route("/process_recording", methods=["POST"])
def process_recording():
    """
    This endpoint returns quickly to Twilio and spawns a background thread
    that performs STT -> LLM -> TTS. We then either update the live call or fallback.
    """
    start_time = time.time()
    try:
        recording_url = request.form.get("RecordingUrl")
        call_sid = request.form.get("CallSid")
        app.logger.info(f"[process_recording] RecordingUrl={recording_url} CallSid={call_sid} form={dict(request.form)}")

        if not recording_url or not call_sid:
            app.logger.error("[process_recording] missing recording_url or call_sid")
            resp = VoiceResponse()
            resp.say("Sorry, an error occurred.", voice="Polly.Joanna", language="en-IN")
            return Response(str(resp), mimetype="text/xml")

        # Spawn background thread (daemon so it won't block process exit)
        thread = threading.Thread(target=async_process_recording, args=(recording_url, call_sid), daemon=True)
        thread.start()
        app.logger.info(f"[process_recording] spawned background thread {thread.name} for call_sid={call_sid}")

        # Immediate response: send caller to /wait which will poll until audio is ready
        resp = VoiceResponse()
        resp.say("Please wait while I prepare your answer...", voice="Polly.Joanna", language="en-IN")
        # redirect to /wait which will pause & re-POST until audio ready
        resp.redirect(f"{Api_Base_url}/wait?call_sid={call_sid}", method="POST")
        elapsed = time.time() - start_time
        app.logger.info(f"[process_recording] returning to Twilio (elapsed {elapsed:.2f}s)")
        return Response(str(resp), mimetype="text/xml")

    except Exception:
        app.logger.exception("[process_recording ERROR]")
        resp = VoiceResponse()
        resp.say("Sorry, an error occurred while processing your request.", voice="Polly.Joanna", language="en-IN")
        return Response(str(resp), mimetype="text/xml")

# === AZURE STT (unchanged) ===
def transcribe_with_azure(audio_path):
    try:
        app.logger.info(f"[STT] transcribing {audio_path}")
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        auto_detect_config = speechsdk.AutoDetectSourceLanguageConfig(languages=["en-IN", "hi-IN", "kn-IN", "mr-IN"])
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config
        )
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = result.properties.get(speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult)
            app.logger.info(f"[STT] recognized text length={len(result.text.strip())} detected_lang={detected}")
            return result.text.strip(), detected
        elif result.reason == speechsdk.ResultReason.NoMatch:
            app.logger.info("[STT] NoMatch")
            return "", ""
        else:
            raise Exception(f"STT failed: {result.reason}")
    except Exception:
        app.logger.exception("[STT ERROR]")
        return "", ""

# === ELEVEN LABS TTS (unchanged except logging) ===
def generate_elevenlabs_audio(text, call_sid=None):
    try:
        app.logger.info(f"[TTS] Generating ElevenLabs audio (len={len(text)})")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
        headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()

        filename = f"{uuid.uuid4().hex}.mp3"
        full_path = os.path.join(STATIC_FOLDER, filename)
        with open(full_path, "wb") as f:
            f.write(r.content)

        if call_sid:
            call_audio_map[call_sid] = filename

        public_base = Api_Base_url
        audio_url = f"{public_base.rstrip('/')}/static/{filename}"
        app.logger.info(f"[TTS] saved file={filename} url={audio_url}")
        return audio_url, filename
    except Exception:
        app.logger.exception("[TTS ERROR]")
        raise

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
        app.logger.info(f"[STATUS] CallSid={call_sid} Status={call_status}")

        if call_sid and call_status in ("completed", "failed", "busy", "no-answer"):
            # cleanup audio file if present
            filename = call_audio_map.pop(call_sid, None)
            if filename:
                file_path = os.path.join(STATIC_FOLDER, filename)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        app.logger.info(f"[STATUS CLEANUP] Deleted audio: {filename}")
                    except Exception:
                        app.logger.exception("[STATUS CLEANUP ERROR] Could not delete file")
            # remove call_info_map entry
            call_info_map.pop(call_sid, None)
        return ("", 204)
    except Exception:
        app.logger.exception("[STATUS ERROR]")
        return ("", 500)

if __name__ == "__main__":
    # debug true for local dev; set false in production
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
