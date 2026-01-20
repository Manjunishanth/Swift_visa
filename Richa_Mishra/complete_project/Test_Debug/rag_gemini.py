# rag_gemini.py
import json
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DATA_DIR = "Data"

index = faiss.read_index(f"{DATA_DIR}/visa_embeddings.index")
vectors = np.load(f"{DATA_DIR}/visa_embeddings.npy")
ids = np.load(f"{DATA_DIR}/visa_ids.npy")

with open(f"{DATA_DIR}/visa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open(f"{DATA_DIR}/visa_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(q):
    return embedder.encode([q], convert_to_numpy=True).astype("float32")

def retrieve_topk(query, k=5):
    q_emb = embed_query(query)
    scores, ids_found = index.search(q_emb, k)

    results = []
    for score, rid in zip(scores[0], ids_found[0]):
        if rid < 0: continue
        uid = str(int(rid))

        results.append({
            "uid": uid,
            "score": float(score),
            "meta": metadata.get(uid, {}),
            "text": chunks.get(uid, "")
        })
    return results

def build_prompt(query, retrieved):
    context = ""
    for i, r in enumerate(retrieved):
        context += f"[{i+1}] {r['text']}\n\n"

    return f"""
Use ONLY the following context to answer the visa question.
If answer is missing -> reply: "Not available in documents".

CONTEXT:
{context}

QUESTION:
{query}

Answer with citations like [1].
"""

def call_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    out = model.generate_content(prompt)
    return out.text

if __name__ == "__main__":
    q = input("Ask your visa question: ")

    retrieved = retrieve_topk(q)
    for r in retrieved:
        print(r)

    prompt = build_prompt(q, retrieved)
    ans = call_gemini(prompt)

    print("\nANSWER:\n")
    print(ans)
