
#updated code Hindi language

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import requests
import openai
import os

app = Flask(__name__)

# Initialize Twilio client
client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# ========== HARDCODED DATABASE ==========
LEADS = {
    "+916203879448": {  # Your test number
        "name": "Test Student",
        "interest": "JEE Preparation",
        "language": "hi",  # 'hi' or 'en'
        "stage": "post-inquiry"
    }
}

COURSES = {
    "hi": {  # Hindi Data
        "JEE": [
            "JEE फाउंडेशन कोर्स (12 महीने, ₹25,000)",
            "JEE एडवांस्ड क्रैश कोर्स (3 महीने, ₹15,000)"
        ],
        "NEET": [
            "NEET अल्टीमेट (18 महीने, ₹30,000)"
        ]
    },
    "en": {  # English Data
        "JEE": [
            "JEE Foundation Course (12 months, ₹25,000)",
            "JEE Advanced Crash Course (3 months, ₹15,000)"
        ],
        "NEET": [
            "NEET Ultimate (18 months, ₹30,000)"
        ]
    }
}
# ========================================

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HINDI_VOICE_ID = "MF3mGyEYCl7XYWbV9V6O"  # Hindi female voice


def text_to_speech(text, language):
    """Convert text to speech using ElevenLabs"""
    if language == "en":
        return None  # Use Twilio's built-in Polly
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_id": HINDI_VOICE_ID,
        "model_id": "eleven_multilingual_v2"
    }
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech",
        headers=headers,
        json=data
    )
    return response.content

@app.route("/answer", methods=["POST"])
def handle_call():
    response = VoiceResponse()
    from_number = request.form.get("From", "")
    
    # Initialize lead if new
    if from_number not in LEADS:
        LEADS[from_number] = {
            "name": "New Lead", 
            "interest": "",
            "language": "hi" if from_number.startswith("+91") else "en",
            "stage": "new"
        }
    
    lead = LEADS[from_number]
    
    # Initial greeting
    if lead["language"] == "hi":
        greeting = "नमस्ते, मैं मेरिटो  दिशा बोल रही हूँ। आप कैसे मदद कर सकती हूँ?"
        audio = text_to_speech(greeting, "hi")
        response.play(audio)
    else:
        response.say("Hello, this is Disha from Meritto. How can I help you?", 
                    voice="Polly.Joanna")
    
    # Start voice input
    gather = Gather(
        input="speech",
        language="hi-IN,en-US",
        action="/process",
        timeout=5
    )
    response.append(gather)
    
    return Response(str(response), mimetype="text/xml")

    except Exception as e:
        print("Error in /answer:", e)
        fallback = VoiceResponse()
        fallback.say("Sorry, an error occurred.", voice="Polly.Joanna")
        return Response(str(fallback), mimetype="text/xml")


@app.route("/process", methods=["POST"])
def process():
    try:
        print("inside /process")
        print("Request Form:", request.form)

        user_input = request.form.get("SpeechResult")
        if not user_input:
            raise ValueError("No speech input received")

        user_input = user_input.lower()
        print("User said:", user_input)

        prompt = "You are Disha from Meritto. Answer in polite English for an education-related query. Keep your answer short and to the point."

        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": f"You are Disha, respond in {'Hindi' if lead['language']=='hi' else 'English'}"
            }, {
                "role": "user",
                "content": user_input
            }]
        ).choices[0].message.content
    
    # Build response
    response = VoiceResponse()
    
    if lead["language"] == "hi":
        audio = text_to_speech(response_text, "hi")
        response.play(audio)
    else:
        response.say(response_text, voice="Polly.Joanna")

        # Continue the conversation
        gather = Gather(
            input="speech dtmf",
            language="en-US",
            action="/process",
            timeout=5
        )
        response.append(gather)

        return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)