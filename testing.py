import numpy as np
from transformers import pipeline

# Define the emotions from the GoEmotions dataset
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# Load the emotion detection model
emotion_model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Initialize an array for the emotions
emotion_array = np.zeros(len(emotions))

def detect_emotions(user_input):
    global emotion_array
    # Detect emotions using the model
    predictions = emotion_model(user_input)
    # Sort predictions by confidence score in descending order
    sorted_predictions = sorted(predictions[0], key=lambda x: x["score"], reverse=True)
    # Select the top 6 emotions
    top_emotions = sorted_predictions[:6]
    
    # Reset the array
    emotion_array = np.zeros(len(emotions))
    
    # Update the array with confidence scores for the top emotions
    for emotion in top_emotions:
        index = emotions.index(emotion["label"])
        emotion_array[index] = emotion["score"]

    return emotion_array

# Example user input
user_input = "i am sad"

# Detect emotions and print the final array
final_emotion_array = detect_emotions(user_input)
print("Emotion Array:")
print(final_emotion_array)

# Print a more readable output
print("\nDetected Emotions with Scores:")
for idx, value in enumerate(final_emotion_array):
    if value > 0:
        print(f"{emotions[idx]}: {value:.2f}")
