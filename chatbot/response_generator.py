import os
import random
import logging
import requests
import pandas as pd
from transformers import pipeline

class ResponseGenerator:
    def __init__(self, csv_path="responses.csv"):
        # Setup logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load responses
        self.responses = self.load_responses(csv_path)
        if not self.responses:
            self.logger.warning("No responses loaded. Please check the CSV file.")

        # Load emotion detection model
        try:
            self.emotion_model = pipeline(
                task="text-classification", 
                model="SamLowe/roberta-base-go_emotions", 
                top_k=None
            )
        except Exception as e:
            self.logger.error(f"Failed to load emotion model: {e}")
            self.emotion_model = None

    def load_responses(self, csv_path):
        """Load responses from a CSV file."""
        try:
            csv_path = os.path.abspath(csv_path)
            data = pd.read_csv(csv_path)
            response_dict = {}
            for _, row in data.iterrows():
                emotion = row["emotion"].strip().lower()
                response = row["response"].strip()
                response_dict.setdefault(emotion, []).append(response)
            return response_dict
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return {}

    def detect_top_two_emotions(self, text):
        """Detect the top two emotions from the input text."""
        try:
            if not self.emotion_model:
                return [("neutral", 0.0), ("neutral", 0.0)]
            predictions = self.emotion_model(text)
            # Sort by confidence score in descending order
            sorted_predictions = sorted(predictions[0], key=lambda x: x["score"], reverse=True)
            # Return top two emotions with their confidence scores
            return [(pred["label"], pred["score"]) for pred in sorted_predictions[:2]]
        except Exception as e:
            self.logger.error(f"Error detecting emotions: {e}")
            return [("neutral", 0.0), ("neutral", 0.0)]

    def get_csv_response(self, emotion):
        """Fetch a response from the loaded CSV based on emotion."""
        return random.choice(self.responses.get(emotion, ["I'm here to listen."]))

    def is_question(self, text):
        """Check if the input text is a question."""
        return text.strip().endswith("?")

    def search_online(self, query):
        """Search online for an answer to the query."""
        try:
            api_key = "your_serpapi_api_key"  # Replace with your actual API key
            base_url = "https://serpapi.com/search"
            params = {
                "q": query,
                "hl": "en",
                "gl": "us",
                "api_key": api_key,
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if "answer_box" in data and "answer" in data["answer_box"]:
                return data["answer_box"]["answer"]
            elif "organic_results" in data and len(data["organic_results"]) > 0:
                return data["organic_results"][0].get("snippet", "No relevant snippet found.")
            else:
                return "I couldn't find any relevant information online."
        except Exception as e:
            self.logger.error(f"Error during online search: {e}")
            return "I'm sorry, I encountered an error while searching. Please try again later."

    def get_response(self, user_message):
        """Generate a response based on the input message."""
        # Check if the input is a question
        if self.is_question(user_message):
            return self.search_online(user_message)

        # Detect the top two emotions
        top_two_emotions = self.detect_top_two_emotions(user_message)
        if len(top_two_emotions) < 2:
            return "I'm here to listen. Can you tell me more?"

        # Compare the two emotions and select the one with the higher confidence
        emotion_1, confidence_1 = top_two_emotions[0]
        emotion_2, confidence_2 = top_two_emotions[1]
        selected_emotion = emotion_1 if confidence_1 >= confidence_2 else emotion_2

        # Fetch a response for the selected emotion
        response = self.get_csv_response(selected_emotion)
        return response
