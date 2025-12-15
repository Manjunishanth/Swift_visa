import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
env_path = "/Users/india/Desktop/visa_chunk_project/.env"
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ---- TEXT GENERATION (Gemini) ----
def generate_text(prompt, model="gemini-1.5-flash"):
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text


# ---- EMBEDDINGS (Gemini) ----
def generate_embedding(text, model="models/text-embedding-004"):
    embedding = genai.embed_content(
        model=model,
        content=text
    )
    return embedding['embedding']


# ---- Generic wrapper for use in other scripts ----
def generic_generate(prompt, provider="gemini", model="gemini-1.5-flash", **kwargs):
    if provider == "gemini":
        return generate_text(prompt, model=model)

    elif provider == "gemini-embedding":
        return generate_embedding(prompt, model=model)

    else:
        raise ValueError("Invalid provider. Use 'gemini' or 'gemini-embedding'.")
