# test_gemini.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()                      # loads .env from current folder
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

resp = genai.embed_content(model="models/text-embedding-004", content="Hello world")
print("Embedding length:", len(resp["embedding"]))
