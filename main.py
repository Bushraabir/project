import os
from fastapi import FastAPI, Request, HTTPException
from chatbot.dialogue_manager import DialogueManager
from utils.logger import setup_logger
import warnings

# Suppress future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize FastAPI app
app = FastAPI()

# Set up logger
logger = setup_logger()

# Define memory CSV file
USER_NAME = "default_user"  # Replace this with dynamic user logic if needed
memory_csv = f"{USER_NAME}_memory.csv"

# Initialize the DialogueManager instance
dialogue_manager = DialogueManager(memory_csv=memory_csv)

@app.get("/")
async def root():
    # Root endpoint to verify that the server is running.
    return {"message": "Welcome to the Mental Health Chatbot"}

@app.post("/chat")
async def chat(request: Request):
    # Endpoint to receive a user message and return a chatbot response.
    try:
        # Parse JSON data from request
        data = await request.json()

        # Validate the user message
        if not isinstance(data, dict) or "message" not in data:
            logger.warning("Invalid request format")
            raise HTTPException(status_code=400, detail="Invalid input. Please send a valid JSON with a 'message' field.")
        
        user_message = data.get("message", "").strip()

        if not user_message:
            logger.warning("Empty message received")
            return {"response": "I'm here to listen. Could you tell me more about how you're feeling?"}
        
        # Log incoming data for debugging
        logger.info(f"Received message: {user_message}")
        
        # Generate a response using the dialogue manager
        response = dialogue_manager.generate_response(user_message)

        # Log generated response
        logger.info(f"Generated response: {response}")

        return {"response": response}

    except HTTPException as http_err:
        # Reraise HTTP exceptions to provide better context to the client
        logger.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )
