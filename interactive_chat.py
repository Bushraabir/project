from chatbot.dialogue_manager import DialogueManager

if __name__ == "__main__":
    print("Welcome to the Mental Health Chatbot. Please enter your name to start chatting.")
    user_name = input("Your Name: ").strip()
    if not user_name:
        user_name = "Guest"

    # Memory CSV for user
    memory_csv = f"{user_name}_memory.csv"

    # Initialize DialogueManager
    dm = DialogueManager(memory_csv=memory_csv)

    print(f"Hello {user_name}, type 'exit' anytime to end the chat.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() == "exit":
            print("Chatbot: Thank you for chatting. Take care!")
            break

        # Generate response
        response = dm.generate_response(user_message)
        print(f"Chatbot: {response}")
