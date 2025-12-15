from dotenv import load_dotenv
import os

load_dotenv()
print("GEMINI KEY:", os.getenv("GEMINI_API_KEY"))
