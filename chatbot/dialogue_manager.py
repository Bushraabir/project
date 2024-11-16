from collections import deque
from datetime import datetime, timedelta
from chatbot.response_generator import ResponseGenerator
from chatbot.emotions import EmotionAnalyzer
from chatbot.nlp_processing import NLPProcessor
import csv
import logging
import requests


class DialogueManager:
    def __init__(self, memory_csv, memory_limit=5, memory_expiry_minutes=10, emotion_threshold=0.1):
        
        
        
        
        self.memory_csv = memory_csv
        self.memory = deque(maxlen=memory_limit)
        self.memory_expiry = timedelta(minutes=memory_expiry_minutes)
        self.emotion_threshold = emotion_threshold

        self.response_generator = ResponseGenerator()
        self.emotion_analyzer = EmotionAnalyzer()
        self.nlp_processor = NLPProcessor()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize memory CSV
        self._initialize_memory_csv()

    def _initialize_memory_csv(self):
        # Create memory CSV if it doesn't exist
        try:
            with open(self.memory_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp", "user_message", "emotion", "chatbot_response"])
        except Exception as e:
            self.logger.error(f"Failed to initialize memory CSV: {e}")

    def detect_emotions(self, text):
        try:
            emotion, confidence = self.emotion_analyzer.analyze(text)
            self.logger.info(f"Detected emotion: {emotion} with confidence: {confidence}")
            return emotion, confidence
        except Exception as e:
            self.logger.error(f"Emotion detection error: {e}")
            return "neutral", 0.0

    def update_memory(self, user_message, emotion, response):
        # Add context to memory and save to CSV
        now = datetime.now()
        self.memory.append({"message": user_message, "emotion": emotion, "response": response, "time": now})
        
        try:
            with open(self.memory_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([now, user_message, emotion, response])
        except Exception as e:
            self.logger.error(f"Failed to save memory to CSV: {e}")

    def generate_response(self, user_message):
        try:
            
            
         # Check if the user is repeating a topic
            if self.memory and self.memory[-1]["message"].lower() == user_message.lower():
                return "It seems like this is important to you. Let’s explore it further."
        
            
            
            # Preprocess user input
            corrected_message = self.nlp_processor.correct_typos(user_message)

        # Detect emotion or query
            emotion, confidence = self.detect_emotions(corrected_message)
            is_question = user_message.strip().endswith("?")

        # If it's a question, search online
            if is_question:
                response = self.response_generator.search_online(corrected_message)
            else:
                # Generate response based on detected emotion
                response = self.response_generator.get_csv_response(emotion)

        # Update memory with user input and chatbot response
            self.update_memory(user_message, emotion, response)

            return response
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "I'm sorry, I encountered an issue. Can you try again?"

    def is_question(self, text):
        return text.strip().endswith("?")


    
    
    
    
    
    def get_memory_context(self):
        # Retrieve the last response based on memory
        if self.memory:
            last_entry = self.memory[-1]
            if datetime.now() - last_entry["time"] < self.memory_expiry:
                return last_entry["response"]
        return None
