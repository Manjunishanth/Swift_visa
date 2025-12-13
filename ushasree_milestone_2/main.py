import os
import re
import json
import numpy as np
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer
from google import genai


# CONFIG

INPUT_DIR = "Data"              # Folder containing PDF/TXT files
CHUNK_SIZE = 300
OVERLAP = 50
EMBED_MODEL_NAME = "all-mpnet-base-v2"
OUTPUT_FAISS = "faiss_index_final.bin"
OUTPUT_EMB = "embeddings_final.npy"
OUTPUT_CHUNKS = "chunks_final.txt"
TOP_K = 5
API_KEY = "Api"  # Replace with your Gemini API key
OUTPUT_JSON = "visa_queries_output.json"

# FUNCTIONS


def pdf_to_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def txt_to_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,:/()\\\- ]+", " ", text)
    return text.strip()

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks

def load_and_process_documents():
    all_chunks = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                raw = pdf_to_text(path)
            elif file.endswith(".txt"):
                raw = txt_to_text(path)
            else:
                continue
            chunks = chunk_text(clean_text(raw))
            all_chunks.extend(chunks)
    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c + "\n")
    return all_chunks

def generate_embeddings(chunks, model_name=EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(chunks)} chunks...")
    emb = model.encode(chunks, batch_size=16, show_progress_bar=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(OUTPUT_EMB, emb)
    print(f"Embeddings saved → {OUTPUT_EMB}")
    return emb

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, OUTPUT_FAISS)
    print(f"FAISS index saved → {OUTPUT_FAISS}")
    print("Total vectors:", index.ntotal)
    return index

def retrieve_top_k(query, k=TOP_K, model_name=EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(OUTPUT_FAISS)
    embeddings = np.load(OUTPUT_EMB)
    with open(OUTPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines()]
    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb)
    distances, indices = index.search(q_emb, k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def ask_gemini(query, retrieved_docs):
    client = genai.Client(api_key=API_KEY)
    context = "\n\n".join(retrieved_docs)
    final_prompt = f"""
You are an experienced US Visa Officer with deep knowledge of:
- H1B, H4, F1, F2, B1/B2, L1, J1, OPT, CPT, EB visas
- USCIS rules, SEVIS, financial requirements, sponsorship rules.

YOUR MANDATORY RULES:
1. First check the retrieved document chunks and use them wherever relevant.should only use the content available in documents.

2. You must ALWAYS give a structured visa-officer style answer:

      ELIGIBILITY: Yes / No / Partially

      REASONS:
      - Bullet 1
      - Bullet 2
      - Bullet 3

      FINAL DECISION:
      - Short visa officer style conclusion.

3. For personal cases (age, income, sponsorship, funding letters):
      → Act like a real visa officer evaluating the case.
      → Use documents.

4. Never invent document content.

---------------------- CONTEXT START ----------------------
{context}
---------------------- CONTEXT END ------------------------

USER QUESTION (Evaluate fully like a visa officer):
{query}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )
        return response.text
    except genai.errors.ClientError as e:
        if 'RESOURCE_EXHAUSTED' in str(e):
            print("Quota exceeded. Please wait and retry later.")
            return "Error: Quota exceeded. Retry later."
        else:
            raise e

# MAIN SCRIPT

if __name__ == "__main__":
    print("\n1. Loading & Processing Documents...")
    chunks = load_and_process_documents()
    print("Total chunks:", len(chunks))

    print("\n2. Generating Embeddings...")
    embeddings = generate_embeddings(chunks)

    print("\n3. Building FAISS Index...")
    build_faiss_index(embeddings)

    # Take multiple queries
    num_queries = int(input("\nHow many visa-related queries do you want to ask? "))
    queries = [input(f"Query {i+1}: ") for i in range(num_queries)]

    results = []
    for q in queries:
        print(f"\nProcessing Query: {q}")
        top_docs = retrieve_top_k(q)
        answer = ask_gemini(q, top_docs)
        results.append({"query": q, "answer": answer})

    # Save results in JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nAll answers saved in {OUTPUT_JSON}")
