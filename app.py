from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from urllib.parse import quote, unquote
import os
import random
from datetime import datetime
from scraper import scrape_lpu_courses
from vector_db import VectorDB
from twilio.rest import Client
import openai  # ✅ Correct import for openai 0.28.x

# ✅ Setup OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
vector_db = VectorDB()

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
        print("Loading courses from LPU website…")
        courses = scrape_lpu_courses()
        if courses:
            vector_db.add_courses(courses)
        else:
            print("Warning: Scraping failed – using existing data")
    else:
        print("Using existing course data")

initialize_courses()

@app.route("/", methods=["GET"])
def health_check():
    return Response("LPU Course Bot is running", content_type="text/plain")

@app.route("/make-call", methods=["POST"])
def make_call():
    data = request.get_json() or {}
    to_number = data.get("phone")
    verified_numbers = ["+919972472457"]
    if not to_number or to_number not in verified_numbers:
        return {"success": False, "message": "Number not verified or missing"}, 400
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    if not all([account_sid, auth_token, from_number]):
        return {"success": False, "message": "Twilio credentials missing"}, 500
    tw_client = Client(account_sid, auth_token)
    call = tw_client.calls.create(
        to=to_number,
        from_=from_number,
        url=os.getenv("CALLBACK_URL"),  # set in env
        machine_detection="Enable",
        status_callback=os.getenv("STATUS_CALLBACK_URL")
    )
    return {"success": True, "sid": call.sid}

@app.route("/answer", methods=["POST"])
def answer_call():
    resp = VoiceResponse()
    resp.pause(length=1)
    resp.say("Hello, this is Disha from LPU.", voice="Polly.Joanna")
    resp.pause(length=0.5)
    resp.say("How may I help you with LPU admissions today?", voice="Polly.Joanna")
    gather = Gather(input="speech", action="/process", language="en-IN", speech_timeout=3)
    resp.append(gather)
    return Response(str(resp), mimetype="text/xml")

@app.route("/process", methods=["POST"])
def process():
    user_input = request.form.get("SpeechResult", "").strip().lower()
    if not user_input:
        return ask_repeat("Could you please repeat that?")
    print(f"Received: {user_input}")
    safe_query = quote(user_input)
    resp = VoiceResponse()
    resp.say(random.choice(FILLER_LINES), voice="Polly.Joanna")
    resp.redirect(f"/wait-response?query={safe_query}")
    return Response(str(resp), mimetype="text/xml")

@app.route("/wait-response", methods=["POST", "GET"])
def wait_response():
    query = unquote(request.args.get("query", "")).lower()
    print(f"Final query: {query}")
    if any(kw in query for kw in ["btech", "engineering", "b.tech"]):
        docs = vector_db.search_courses("btech engineering")
        if docs:
            msg = generate_btech_response(docs)
            return voice_response(msg)
    gen = generate_general_response(query)
    return voice_response(gen)

def generate_btech_response(courses):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You’re Disha from LPU. Mention 2‑3 key programs: {courses}. Under 15 words, end with a question."},
            {"role": "user", "content": "What B.Tech programs does LPU offer?"}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content

def generate_general_response(query):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You’re Disha, an LPU admission counselor. Friendly Indian English (1 sentence)."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content

def voice_response(text):
    vr = VoiceResponse()
    vr.say(text, voice="Polly.Joanna")
    vr.append(Gather(input="speech", action="/process", speech_timeout=2))
    return Response(str(vr), mimetype="text/xml")

def fallback_response(msg):
    vr = VoiceResponse()
    vr.say(msg, voice="Polly.Joanna")
    vr.redirect("/answer")
    return Response(str(vr), mimetype="text/xml")

def ask_repeat(msg):
    vr = VoiceResponse()
    vr.say(msg, voice="Polly.Joanna")
    vr.redirect("/answer")
    return Response(str(vr), mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
