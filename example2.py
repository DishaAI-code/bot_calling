from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import requests
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ========== HARDCODED LEAD DATA ==========

LEADS = {
    "+916203879448": {
        "name": "Test Student",
        "interest": "JEE Preparation",
        "language": "hi",
        "stage": "post-inquiry"
    }
}

COURSES = {
    "hi": {
        "JEE": [
            "JEE फाउंडेशन कोर्स (12 महीने, ₹25,000)",
            "JEE एडवांस्ड क्रैश कोर्स (3 महीने, ₹15,000)"
        ],
        "NEET": [
            "NEET अल्टीमेट (18 महीने, ₹30,000)"
        ]
    },
    "en": {
        "JEE": [
            "JEE Foundation Course (12 months, ₹25,000)",
            "JEE Advanced Crash Course (3 months, ₹15,000)"
        ],
        "NEET": [
            "NEET Ultimate (18 months, ₹30,000)"
        ]
    }
}

# ========== API CONFIG ==========

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HINDI_VOICE_ID = "MF3mGyEYCl7XYWbV9V6O"

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")

openai.api_key = os.getenv("OPENAI_API_KEY")

# ========== UTILITY FUNCTIONS ==========

def text_to_speech(text, language):
    if language == "en":
        return None

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

def translate_text_azure(text, to_lang="en"):
    url = f"{AZURE_TRANSLATOR_ENDPOINT}translate?api-version=3.0&to={to_lang}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json"
    }
    body = [{"text": text}]
    response = requests.post(url, headers=headers, json=body)
    result = response.json()
    return result[0]["translations"][0]["text"]

# ========== ROUTES ==========

@app.route("/answer", methods=["POST"])
def handle_call():
    response = VoiceResponse()
    from_number = request.form.get("From", "")

    if from_number not in LEADS:
        LEADS[from_number] = {
            "name": "New Lead",
            "interest": "",
            "language": "hi" if from_number.startswith("+91") else "en",
            "stage": "new"
        }

    lead = LEADS[from_number]
    language = lead["language"]

    if language == "hi":
        greeting = "नमस्ते, मैं मेरिटो दिशा बोल रही हूँ। आप कैसे मदद कर सकती हूँ?"
        audio = text_to_speech(greeting, "hi")
        response.play(audio)
    else:
        response.say("Hello, this is Disha from Meritto. How can I help you?", voice="Polly.Joanna")

    gather = Gather(
        input="speech",
        language="hi-IN",  # Focused on Hindi only
        action="/process",
        timeout=6,
        actionOnEmptyResult=True
    )
    response.append(gather)

    return Response(str(response), mimetype="text/xml")

@app.route("/process", methods=["POST"])
def process():
    from_number = request.form.get("From", "")
    user_input = request.form.get("SpeechResult", "")
    lead = LEADS.get(from_number, {"language": "hi"})

    print("📞 From:", from_number)
    print("🎤 SpeechResult:", repr(user_input))

    language = lead["language"]
    response = VoiceResponse()

    if not user_input or user_input.strip() == "":
        print("⚠️ No speech detected.")
        if language == "hi":
            audio = text_to_speech("माफ़ कीजिए, मैं आपको समझ नहीं पाई। कृपया दोबारा बोलें।", "hi")
            response.play(audio)
        else:
            response.say("Sorry, I didn't catch that. Please say that again.", voice="Polly.Joanna")

        gather = Gather(
            input="speech",
            language="hi-IN",
            action="/process",
            timeout=6,
            actionOnEmptyResult=True
        )
        response.append(gather)
        return Response(str(response), mimetype="text/xml")

    try:
        print("🌐 Step 1: Translating Hindi to English...")
        translated_input = translate_text_azure(user_input, to_lang="en")
        print("🔁 Translated to English:", translated_input)

        print("🤖 Step 2: Sending to OpenAI GPT...")
        gpt_response_en = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Disha from Meritto. Respond in English."},
                {"role": "user", "content": translated_input}
            ]
        ).choices[0].message.content
        print("🧠 GPT Response (EN):", gpt_response_en)

        print("🌐 Step 3: Translating English to Hindi...")
        response_text = translate_text_azure(gpt_response_en, to_lang="hi")
        print("✅ Final Response (HI):", response_text)

    except Exception as e:
        print("❌ Error during translation or GPT:", e)
        response_text = "माफ़ कीजिए, कुछ तकनीकी समस्या हो गई है। कृपया बाद में प्रयास करें।"

    if language == "hi":
        audio = text_to_speech(response_text, "hi")
        response.play(audio)
    else:
        response.say(response_text, voice="Polly.Joanna")

    gather = Gather(
        input="speech",
        language="hi-IN",
        action="/process",
        timeout=6,
        actionOnEmptyResult=True
    )
    response.append(gather)

    return Response(str(response), mimetype="text/xml")

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
