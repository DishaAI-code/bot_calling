from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from openai import OpenAI  # Updated import
import os
from datetime import datetime
from scraper import scrape_lpu_courses
from vector_db import VectorDB

app = Flask(__name__)

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Updated initialization
vector_db = VectorDB()

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
            return ask_repeat("Could you please repeat that?")

        print(f"Processing query: {user_input}")

        # Enhanced course detection
        if any(kw in user_input for kw in ["btech", "engineering", "b.tech"]):
            courses = vector_db.search_courses("btech engineering", n_results=5)
            if courses:
                response = generate_btech_response(courses)
                return voice_response(response)

        # General query fallback
        response = generate_general_response(user_input)
        return voice_response(response)

    except Exception as e:
        print("Processing error:", e)
        return fallback_response("Let me transfer you to our admission desk for assistance.")

# AI Response Generators (Updated for OpenAI v1.0+)
def generate_btech_response(courses):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""You're Disha from LPU. Respond to B.Tech queries with:
- Friendly Indian English
- Mention 2-3 key programs from: {courses}
- Keep response under 15 words
- End with a question"""},
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

# Response Helpers
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



#--------------------------------------------------date - 16th ------------------

# from flask import Flask, request, Response
# from twilio.twiml.voice_response import VoiceResponse, Gather
# import openai
# import os
# from dotenv import load_dotenv

# app = Flask(__name__)

# # Set your OpenAI API Key (if not already in environment)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# @app.route("/", methods=["GET"])
# def health_check():
#     return Response("this is base testing url", content_type="text/plain")

# @app.route("/answer", methods=["POST"])
# def answer_call():
#     try:
#         print("inside /answer")
#         response = VoiceResponse()
#         response.say("Hello, this is Disha calling from LPU. How can I assist you today?", voice="Polly.Joanna")

#         gather = Gather(
#             input="speech dtmf",
#             language="en-US",
#             action="/process",
#             timeout=5
#         )
#         response.append(gather)

#         return Response(str(response), mimetype="text/xml")

#     except Exception as e:
#         print("Error in /answer:", e)
#         fallback = VoiceResponse()
#         fallback.say("Sorry, an error occurred.", voice="Polly.Joanna")
#         return Response(str(fallback), mimetype="text/xml")


# @app.route("/process", methods=["POST"])
# def process():
#     try:
#         print("inside /process")
#         print("Request Form:", request.form)

#         user_input = request.form.get("SpeechResult")
#         if not user_input:
#             raise ValueError("No speech input received")

#         user_input = user_input.lower()
#         print("User said:", user_input)

#         prompt = "You are Disha from Meritto. Answer in polite English for an education-related query. Keep your answer short and to the point."

#         chat_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": user_input}
#             ],
#             timeout=10
#         )

#         response_text = chat_response.choices[0].message['content']
#         print("Chat response:", response_text)

#         response = VoiceResponse()
#         response.say(response_text, voice="Polly.Joanna")

#         # Continue the conversation
#         gather = Gather(
#             input="speech dtmf",
#             language="en-US",
#             action="/process",
#             timeout=5
#         )
#         response.append(gather)

#         return Response(str(response), mimetype="text/xml")

#     except Exception as e:
#         print("Error in /process:", e)
#         fallback = VoiceResponse()
#         fallback.say("Sorry, an error occurred while processing your request.", voice="Polly.Joanna")
#         return Response(str(fallback), mimetype="text/xml")


# # Optional: To handle Twilio call status updates (required if used in call.py)
# @app.route("/status", methods=["POST"])
# def call_status():
#     print("Call status:", request.form.to_dict())
#     return ("", 204)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
