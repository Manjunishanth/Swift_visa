import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=API_KEY)

CHUNKS_DIR = "chunks"
OUT_DIR = "embeddings_output"
os.makedirs(OUT_DIR, exist_ok=True)

model = "models/embedding-001"

for pdf_folder in os.listdir(CHUNKS_DIR):
    pdf_path = os.path.join(CHUNKS_DIR, pdf_folder)
    if not os.path.isdir(pdf_path):
        continue

    embeddings = []

    for file in os.listdir(pdf_path):
        if file.endswith(".txt"):
            text = open(os.path.join(pdf_path, file)).read()

            emb = genai.embed_content(
                model=model,
                content=text
            )["embedding"]

            embeddings.append({
                "chunk": file,
                "text": text,
                "embedding": emb
            })

    out_file = os.path.join(OUT_DIR, f"{pdf_folder}.json")
    with open(out_file, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Saved JSON embeddings → {out_file}")
