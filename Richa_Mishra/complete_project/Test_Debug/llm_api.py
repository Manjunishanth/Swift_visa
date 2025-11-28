from mistralai import Mistral
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Read API key
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize client
client = Mistral(api_key=api_key)

model = "mistral-tiny"  # or "mistral-small"

topic = input("Enter a topic you want explained: ")

response = client.chat.complete(
    model=model,
    messages=[
        {"role": "system", "content": "you are a helpful tutor who explains concepts simply."},
        {"role": "user", "content": f"Explain {topic} in 4-5 sentences."} # roles can be 'user', 'assistant', or 'system'
    ]
)

# Correct way to print content
print("\n=== Mistral Response ===")
print(topic,"\n")
print(response.choices[0].message.content)





# from mistralai import Mistral
# from dotenv import load_dotenv
# import os

# # Load .env
# load_dotenv()
# api_key = os.getenv("MISTRAL_API_KEY")

# # Initialize client
# client = Mistral(api_key=api_key)
# model = "mistral-tiny"

# # Start conversation history
# messages = [
#     {"role": "system", "content": "You are a helpful tutor who explains concepts simply."}
# ]

# while True:
#     # Get user input
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Conversation ended.")
#         break

#     # Add user message to history
#     messages.append({"role": "user", "content": user_input})

#     # Get assistant response
#     response = client.chat.complete(
#         model=model,
#         messages=messages
#     )

#     # Extract assistant reply
#     assistant_reply = response.choices[0].message.content
#     print(f"LLM: {assistant_reply}")

#     # Add assistant reply to history
#     messages.append({"role": "assistant", "content": assistant_reply})