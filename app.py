

# # 3    --- upddation

# from flask import Flask, request, Response
# from twilio.twiml.voice_response import VoiceResponse, Gather
# import openai

# app = Flask(__name__)

# # Hardcoded English Course Data
# # COURSES = {
# #     "jee": {
# #         "name": "jee",
# #         "programs": [
# #             "JEE Foundation Course: 12 months duration, ₹25,000 fee",
# #             "JEE Advanced Crash Course: 3 months duration, ₹15,000 fee"
# #         ],
# #         "keywords": ["jee", "engineering", "iit", "joint entrance"]
# #     },
# #     "neet": {
# #         "name": "neet",
# #         "programs": [
# #             "NEET Ultimate Program: 18 months duration, ₹30,000 fee"
# #         ],
# #         "keywords": ["neet", "medical", "aiims", "doctor"]
# #     }
# # }

# @app.route("/", methods=["GET"])
# def health_check():
#     return Response("this is base testing url ", content_type="text/plain")

# @app.route("/answer", methods=["POST"])
# def answer_call():
#     try:
#         print("inside try block")
#         response = VoiceResponse()
#         response.say("Hello, this is Disha calling from Lpu. How i can i assist you Today ", voice="Polly.Joanna")

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
#         # return Response(str(fallback), mimetype="text/xml")
#         return Response("hellow world")

# @app.route("/process", methods=["POST"])
# def process():
#     try:
#         user_input = request.form.get("SpeechResult", "").lower()
#         print("User said:", user_input)

#         prompt = "You are Disha from Meritto. Answer in polite English for an education-related query. Keep your Answer Short and To the point"
#         chat_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": user_input}
#             ]
#         )
#         response_text = chat_response.choices[0].message['content']
#         print("Chat response:", response_text)

#         response = VoiceResponse()
#         response.say(response_text, voice="Polly.Joanna")

#         # Continue conversation
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
#         fallback.say("Sorry, an error occurred.", voice="Polly.Joanna")
#         return Response(str(fallback), mimetype="text/xml")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



# from flask import Flask, request, Response
# from twilio.twiml.voice_response import VoiceResponse, Gather
# import openai

# app = Flask(__name__)

# # Hardcoded English Course Data
# COURSES = {
#     "JEE": [
#         "JEE Foundation Course, 12 months, ₹25,000",
#         "JEE Advanced Crash Course, 3 months, ₹15,000"
#     ],
#     "NEET": [
#         "NEET Ultimate, 18 months, ₹30,000"
#     ]
# }

# @app.route("/", methods=["GET"])
# def health_check():
#     return Response("Meritto Voicebot is live (English-only mode)", content_type="text/plain")

# @app.route("/answer", methods=["POST"])
# def answer_call():
#     try:
#         print("inside try block")
#         response = VoiceResponse()
#         response.say("Hello, this is Disha from Meritto. Please tell me if you're interested in JEE or NEET.", voice="Polly.Joanna")

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
#         user_input = request.form.get("SpeechResult", "").lower()
#         digit_input = request.form.get("Digits", "")
#         response_text = ""

#         # Determine based on either speech or digit
#         if "jee" in user_input or digit_input == "1":
#             response_text = " ".join(COURSES["JEE"])
#         elif "neet" in user_input or digit_input == "2":
#             response_text = " ".join(COURSES["NEET"])
#         else:
#             prompt = "You are Disha from Meritto. Answer in polite English for an education-related query."
#             chat_response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": prompt},
#                     {"role": "user", "content": user_input}
#                 ]
#             )
#             response_text = chat_response.choices[0].message['content']

#         response = VoiceResponse()
#         response.say(response_text, voice="Polly.Joanna")

#         # Add gather again for follow-up
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
#         fallback.say("Sorry, an error occurred.", voice="Polly.Joanna")
#         return Response(str(fallback), mimetype="text/xml")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



#--------------------------------------------------date - 16th ------------------

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import openai
import os

app = Flask(__name__)

# Set your OpenAI API Key (if not already in environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=["GET"])
def health_check():
    return Response("this is base testing url", content_type="text/plain")

@app.route("/answer", methods=["POST"])
def answer_call():
    try:
        print("inside /answer")
        response = VoiceResponse()
        response.say("Hello, this is Disha calling from LPU. How can I assist you today?", voice="Polly.Joanna")

        gather = Gather(
            input="speech dtmf",
            language="en-US",
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
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            timeout=10
        )

        response_text = chat_response.choices[0].message['content']
        print("Chat response:", response_text)

        response = VoiceResponse()
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

    except Exception as e:
        print("Error in /process:", e)
        fallback = VoiceResponse()
        fallback.say("Sorry, an error occurred while processing your request.", voice="Polly.Joanna")
        return Response(str(fallback), mimetype="text/xml")


# Optional: To handle Twilio call status updates (required if used in call.py)
@app.route("/status", methods=["POST"])
def call_status():
    print("Call status:", request.form.to_dict())
    return ("", 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
