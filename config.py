import os

# Environment configuration
CHATBOT_BASE_URL = os.getenv("CHATBOT_BASE_URL", "http://127.0.0.1:8000/chat")
INTENT_MODEL_PATH = "models/intent_model"
EMOTION_MODEL_PATH = "models/emotion_model"
PORT = int(os.getenv("PORT", 8000))