# chatbot/nlp_processing.py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

nltk.download("punkt")
nltk.download("stopwords")

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> list:
        tokens = word_tokenize(text)
        return [token for token in tokens if token.isalnum() and token.lower() not in self.stop_words]

    def correct_typos(self, text: str) -> str:
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text
