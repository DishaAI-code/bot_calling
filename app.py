from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from urllib.parse import quote, unquote
import os
import random
from datetime import datetime
from scraper import scrape_lpu_courses
from vector_db import VectorDB
from twilio.rest import Client
from openai import OpenAI

# ✅ Initialize OpenAI client (latest SDK format)
client = OpenAI()

app = Flask(__name__)

# Initialize vector DB
vector_db = VectorDB()

# Random filler lines to reduce perceived lag
FILLER_LINES = [
    "Hmm, let me check that for you.",
    "One sec, I'm fetching the info.",
    "Alright, give me a moment to look that up.",
    "Okay, just a second please.",
    "Let me pull that up for you.",
    "Let me find the best answer for you."
]

def initialize_courses():
    if not vector_db.has_courses() or datetime.now().weekday() == 0:
        print("Loading courses from LPU website...")
        courses = scrape_lpu_courses()
        if courses:
            vector_db.add_courses(courses)
        else:
            print("Warning: Scraping failed - using existing data")
    else:
        print("Using existing course data")

initialize_courses()

@app.route("/", methods=["GET"])
def health_check():
    return Response("LPU Course Bot is running", content_type="text/plain")

@app.route("/make-call", methods=["POST"])
def make_call():
    try:
        data = request.get_json()
        to_number = data.get("phone")

        # ✅ Optional: Limit to verified numbers if you're on trial
        verified_numbers = ["+919972472457"]
        if not to_number or to_number not in verified_numbers:
            return {"success": False, "message": "Number not verified or missing"}, 400

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_PHONE_NUMBER")

        if not all([account_sid, auth_token, from_number]):
            return {"success": False, "message": "Twilio credentials missing"}, 500

        client_twilio = Client(account_sid, auth_token)

        call = client_twilio.calls.create(
            to=to_number,
            from_=from_number,
            url="https://web-production-e1e66.up.railway.app/answer",
            machine_detection="Enable",
            status_callback="https://web-production-e1e66.up.railway.app/status"
        )

        return {"success": True, "sid": call.sid}

    except Exception as e:
        print("[/make-call] Error:", e)
        return {"success": False, "message": str(e)}, 500

@app.route("/answer", methods=["POST"])
def answer_call():
    try:
        response = VoiceResponse()
        response.pause(length=1)
        response.say("Hello, this is Disha from LPU.", voice="Polly.Joanna")
        response.pause(length=0.5)
        response.say("How may I help you with LPU admissions today?", voice="Polly.Joanna")

        gather = Gather(
            input="speech",
            action="/process",
            language="en-IN",
            speech_timeout=3,
            speech_model="experimental_conversations"
        )
        response.append(gather)

        return Response(str(response), mimetype="text/xml")

    except Exception as e:
        print("Error in /answer:", e)
        return fallback_response("Please hold while we connect you to an advisor.")

@app.route("/process", methods=["POST"])
def process():
    try:
        user_input = request.form.get("SpeechResult", "").strip().lower()
        if not user_input:
            print("[/process] Empty user input")
            return ask_repeat("Could you please repeat that?")

        print(f"[/process] Received: {user_input}")
        safe_query = quote(user_input)

        filler_line = random.choice(FILLER_LINES)

        response = VoiceResponse()
        response.say(filler_line, voice="Polly.Joanna")
        response.redirect(f"/wait-response?query={safe_query}")
        return Response(str(response), mimetype="text/xml")

    except Exception as e:
        print("[/process] Error:", e)
        return fallback_response("Let me transfer you to our admission desk for assistance.")

@app.route("/wait-response", methods=["POST", "GET"])
def wait_response():
    try:
        raw_query = request.args.get("query", "")
        query = unquote(raw_query).lower()

        print(f"[/wait-response] Final query: {query}")

        if any(kw in query for kw in ["btech", "engineering", "b.tech"]):
            courses = vector_db.search_courses("btech engineering", n_results=5)
            if courses:
                message = generate_btech_response(courses)
                return voice_response(message)

        message = generate_general_response(query)
        return voice_response(message)

    except Exception as e:
        print("[/wait-response] Error:", e)
        return fallback_response("Apologies, let me transfer you to the admissions desk.")

# ---------- AI Response Generators ----------

def generate_btech_response(courses):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""You're Disha from LPU. Respond to B.Tech queries with:
- Friendly Indian English
- Mention 2-3 key programs from: {courses}
- Keep response under 15 words
- End with a question"""
            },
            {"role": "user", "content": "What B.Tech programs does LPU offer?"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_general_response(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're Disha, an LPU admission counselor. Respond in friendly Indian English (1 sentence)."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------- Helpers ----------

def voice_response(text):
    vr = VoiceResponse()
    vr.say(text, voice="Polly.Joanna")
    gather = Gather(
        input="speech",
        action="/process",
        speech_timeout=2
    )
    vr.append(gather)
    return Response(str(vr), mimetype="text/xml")

def fallback_response(message):
    vr = VoiceResponse()
    vr.say(message, voice="Polly.Joanna")
    vr.redirect("/answer")
    return Response(str(vr), mimetype="text/xml")

def ask_repeat(message):
    vr = VoiceResponse()
    vr.say(message, voice="Polly.Joanna")
    vr.redirect("/answer")
    return Response(str(vr), mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
