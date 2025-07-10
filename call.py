
# from twilio.rest import Client
# from dotenv import load_dotenv
# import os

# load_dotenv()

# client = Client(
#     os.getenv("TWILIO_ACCOUNT_SID"),
#     os.getenv("TWILIO_AUTH_TOKEN")
# )

# def make_call(to_number):
#     call = client.calls.create(
#         url=f"{os.getenv('WEBHOOK_URL')}/answer",
#         to=to_number,
#         from_=os.getenv("TWILIO_PHONE_NUMBER"),
#         record=False
#     )
#     print(f"Calling {to_number}... SID: {call.sid}")

# if __name__ == "__main__":
#     make_call("+916203879448")  # Replace with your number

from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

def make_call(to_number):
    call = client.calls.create(
        url=f"{os.getenv('WEBHOOK_URL')}/answer",
        to=to_number,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        record=True,  # ← Must be True for Hindi STT
        recording_channels='dual',  # Records both parties
        recording_status_callback=f"{os.getenv('WEBHOOK_URL')}/recording-events",
        machine_detection="Enable"  # Optional but recommended
    )
    print(f"Call initiated to {to_number}. SID: {call.sid}")

if __name__ == "__main__":
    # Test with Indian number format
    make_call("+916203879448")  # Keep this format (+91 followed by 10 digits)