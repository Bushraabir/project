from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
import torch
from datasets import Dataset

# ---------- Configuration ----------
MODEL_NAME = "distilbert-base-uncased"  # Pre-trained model name
MODEL_PATH = "./trained_model"         # Path to save the trained model
NUM_LABELS = 6                         # Number of emotion classes
NUM_EPOCHS = 3                         # Training epochs
BATCH_SIZE = 8                         # Batch size

# ---------- Load Dataset ----------
def load_emotion_dataset():
    """
    Load a dataset for training. Options:
    - Hugging Face datasets (e.g., "emotion", "imdb")
    - Custom CSV files converted to Hugging Face Dataset.
    """
    try:
        # Attempt to load a Hugging Face dataset
        print("Loading Hugging Face 'emotion' dataset...")
        # dataset = load_dataset("emotion")
    except Exception as e:
        # If Hugging Face dataset fails, fallback to CSV
        print("Hugging Face dataset failed. Loading custom CSV...")
        data = pd.read_csv("path_to_your_emotion_dataset.csv")  # Ensure this file exists
        dataset = Dataset.from_pandas(data)
    
    return dataset

dataset = load_emotion_dataset()

# ---------- Tokenize Dataset ----------
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ---------- Split Dataset ----------
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# ---------- Load Model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# ---------- Training Arguments ----------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# ---------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ---------- Train the Model ----------
print("Training the model...")
trainer.train()

# Save the trained model
print("Saving the model...")
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

# ---------- Test the Model ----------
def test_model(text: str):
    """
    Test the trained model with a sample input text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    # Get emotion label (index of the highest score)
    predicted_emotion = torch.argmax(outputs.logits, dim=1).item()
    print(f"Predicted emotion: {predicted_emotion}")

# Test the model with a sample input
test_text = "I feel very sad today."
print(f"Testing model with input: '{test_text}'")
test_model(test_text)
