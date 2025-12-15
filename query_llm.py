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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print("✅ Gemini API key loaded")

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(BASE_DIR, "chunks_out", "USA_Visa_Screening_Details")
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
chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.startswith("chunk_")])

texts = []
for file in chunk_files:
    with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
        texts.append(f.read())

print(f"✅ Loaded {len(texts)} chunks")

# =============================
# USER INPUT (ONLY ONCE)
# =============================
query = input("\n🔎 Ask your visa question: ").strip()

# =============================
# Similarity search (TF-IDF)
# =============================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts + [query])

query_vec = tfidf_matrix[-1]
doc_vecs = tfidf_matrix[:-1]

similarities = cosine_similarity(query_vec, doc_vecs)[0]
best_index = np.argmax(similarities)
best_chunk = texts[best_index]

print("\n📌 Most relevant chunk selected")
print("-" * 50)
print(best_chunk)

# =============================
# Gemini Answer (ONCE)
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
