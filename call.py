# from twilio.rest import Client
# from dotenv import load_dotenv
# import os
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# # Initialize Twilio client
# client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

# twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
# def make_call(to_number):
#     """Make call to verified numbers only"""
#     verified_numbers = ["+916203879448"]  # Your verified numbers
    
#     if to_number not in verified_numbers:
#         logger.error(f"Cannot call unverified number: {to_number}")
#         print(" Trial accounts can only call verified numbers. Add this number in Twilio console.")
#         return

#     try:
#         call = client.calls.create(
#             url=f"{os.getenv('WEBHOOK_URL')}/answer",
#             to=to_number,
#             from_=twilio_number,
#             record=False,
#             machine_detection="Enable",
#             status_callback=f"{os.getenv('WEBHOOK_URL')}/status"
#         )
#         logger.info(f"Call initiated to {to_number} | SID: {call.sid}")
#         print("webhook url is ",call.uri)
#         print(f"🚀 Calling {to_number} | Status: {call.status}")
#         print("twilio number is ",twilio_number)
#     except Exception as e:
#         logger.error(f"Call failed: {e}")
#         print(f"❌ Call failed: {str(e)}")

# if __name__ == "__main__":
#     # Call your verified number
#     make_call("+916203879448")  # Replace with your actual verified number




# from twilio.rest import Client
# from dotenv import load_dotenv
# import os
# import time

# # Load environment variables
# load_dotenv()

# # Initialize Twilio client
# client = Client(
#     os.getenv("TWILIO_ACCOUNT_SID"),
#     os.getenv("TWILIO_AUTH_TOKEN")
# )

# def make_call(to_number, record_call=False):
#     """Make an outbound call with error handling"""
#     try:
#         call = client.calls.create(
#             url=f"{os.getenv('WEBHOOK_URL')}/answer",  # From .env
#             to=to_number,
#             from_=os.getenv("TWILIO_PHONE_NUMBER"),
#             record=record_call,
#             status_callback=f"{os.getenv('WEBHOOK_URL')}/status",  # Optional callback
#             machine_detection="Enable"  # Detect answering machines
#         )
        
#         print(f"🚀 Call initiated to {to_number}")
#         print(f"📞 Call SID: {call.sid}")
#         print(f"⏱️ Call status: {call.status}")
        
#         return call.sid
#     except Exception as e:
#         print(f"❌ Call failed: {e}")
#         return None

# if __name__ == "__main__":
#     # Example usage
#     number_to_call = "+916203879448"  # Replace with actual number
#     call_sid = make_call(number_to_call, record_call=True)
    
#     if call_sid:
#         print("✅ Call successfully initiated!")
#     else:
#         print("❌ Failed to initiate call")



# ----------------------------------------------- date = 16th---------------------

from twilio.rest import Client
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Twilio client
client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
webhook_url = os.getenv("WEBHOOK_URL")  # Make sure it's public & https

def make_call(to_number):
    verified_numbers = ["+916203879448"]  # Add verified numbers here

    if to_number not in verified_numbers:
        logger.error(f"Cannot call unverified number: {to_number}")
        print("❌ Trial accounts can only call verified numbers.")
        return

    try:
        call = client.calls.create(
            url="https://4f745b8969c7.ngrok-free.app/answer", 
            to=to_number,
            from_=twilio_number,
            record=False,
            machine_detection="Enable",
            status_callback="https://4f745b8969c7.ngrok-free.app/status"
        )

        logger.info(f"✅ Call initiated to {to_number} | SID: {call.sid}")
        print("📞 Call URI:", call.uri)
        print("Twilio number:", twilio_number)

    except Exception as e:
        logger.error(f"❌ Call failed: {e}")
        print(f"❌ Call failed: {str(e)}")


if __name__ == "__main__":
    make_call("+916203879448")
