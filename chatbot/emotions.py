
from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        # Initialize the EmotionAnalyzer with the specified model.
        try:
            self.emotion_pipeline = pipeline("text-classification", model=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize emotion pipeline: {e}")

    def analyze(self, text):
        try:
            prediction = self.emotion_pipeline(text)
            if prediction:
                return prediction[0]["label"].lower(), prediction[0]["score"]
            return "neutral", 0.0
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return "neutral", 0.0


    def analyze_batch(self, texts):
        # Analyze emotions for a batch of texts.
        if not isinstance(texts, list) or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Input must be a list of non-empty strings.")

        try:
            predictions = self.emotion_pipeline(texts)
            return [pred["label"].lower() if pred else "neutral" for pred in predictions]
        except Exception as e:
            print(f"Error during batch emotion analysis: {e}")
            return ["neutral"] * len(texts)
