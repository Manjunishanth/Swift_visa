# test_gemini.py

import os
from google import genai  

def main():
    # Read API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Initialize client
    client = genai.Client()

    # Send a simple prompt
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello Gemini! Please say hello if my API key works."
    )

    print("--- Response from Gemini ---")
    print(response.text)

if __name__ == "__main__":
    main()
