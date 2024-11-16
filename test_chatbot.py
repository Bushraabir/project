# project/test_chatbot.py
import requests
from config import CHATBOT_BASE_URL

payload = {"message": "I feel sad today"}

response = requests.post(CHATBOT_BASE_URL, json=payload)
print("Response from chatbot:", response.json())
