import os
import json
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

print("🔹 Query script started")

# =============================
# Load API Key
# =============================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
print("✅ Gemini API key loaded")

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n🌍 Available Countries:")
print("1. USA")
print("2. UK")
print("3. Canada")
print("4. Schengen")
print("5. Ireland")

choice = input("Select country (1-5): ").strip()

country_map = {
    "1": "USA_Visa_Screening_Details",
    "2": "UK_Visa_Screening_Details",
    "3": "Canada_Visa_Screening_Details",
    "4": "Schengen_Visa_Screening_Details",
    "5": "Ireland_Visa_Screening_Details"
}

if choice not in country_map:
    print("❌ Invalid selection")
    exit()

CHUNK_DIR = os.path.join(
    BASE_DIR,
    "chunks_out",
    country_map[choice]
)

EMBEDDINGS_PATH = os.path.join(CHUNK_DIR, "embeddings.json")


# =============================
# Load embeddings
# =============================
with open(EMBEDDINGS_PATH, "r") as f:
    data = json.load(f)

embeddings = np.array(data["embeddings"])
print(f"✅ Loaded {len(embeddings)} embeddings")

# =============================
# Load chunk texts
# =============================
chunk_files = sorted(
    [f for f in os.listdir(CHUNK_DIR) if f.startswith("chunk_")]
)

texts = []
for file in chunk_files:
    with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
        texts.append(f.read())

print(f"✅ Loaded {len(texts)} chunks")

# =============================
# Ask Question
# =============================
query = input("\n🔎 Ask your visa question: ")

# =============================
# TF-IDF similarity (NO API usage)
# =============================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts + [query])

query_vec = tfidf_matrix[-1]
doc_vecs = tfidf_matrix[:-1]

similarities = cosine_similarity(query_vec, doc_vecs)[0]
top_k = 3
top_indices = similarities.argsort()[-top_k:][::-1]

best_chunk = "\n\n".join([texts[i] for i in top_indices])


print("\n📌 Most relevant chunk selected")
print("-" * 50)
print(best_chunk)

# =============================
# Gemini Answer
# =============================
print("\n🤖 Generating final answer using Gemini...")

model = genai.GenerativeModel("models/gemini-flash-latest")

prompt = f"""
Answer the question using ONLY the information below.

Context:
{best_chunk}

Question:
{query}
"""

response = model.generate_content(prompt)

print("\n✅ Final Answer:\n")
print(response.text)
