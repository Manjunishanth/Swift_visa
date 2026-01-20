import os
import re
import numpy as np
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer


# CONFIG

INPUT_DIR = "Data"
CHUNK_SIZE = 300
OVERLAP = 50
EMBED_MODEL_NAME = "all-mpnet-base-v2"

OUTPUT_FAISS_TEST = "faiss_index.bin"
OUTPUT_EMB_TEST = "embeddings.npy"


# 1. PREPROCESSING (PDF/TXT → TEXT)

def pdf_to_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + " "
    return text


def txt_to_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text):
    text = re.sub(r"\s+", " ", text)                       
    text = re.sub(r"[^A-Za-z0-9.,:/()\- ]+", " ", text)    
    return text.strip()


# 2. CHUNKING

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + size])
        chunks.append(chunk)
        start += size - overlap

    return chunks


# LOAD + PROCESS ALL DOCUMENTS

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

            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)

    return all_chunks


# 3. EMBEDDING

def generate_embeddings(chunks, model_name=EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    print(f"[+] Embedding {len(chunks)} chunks...")

    emb = model.encode(chunks, batch_size=16, show_progress_bar=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    np.save(OUTPUT_EMB_TEST, emb)
    print(f"[✓] Embeddings saved → {OUTPUT_EMB_TEST}")
    print("Embedding shape:", emb.shape)

    return emb


# 4. FAISS INDEX

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    print("[+] Building FAISS index...")
    index.add(embeddings)

    faiss.write_index(index, OUTPUT_FAISS_TEST)

    print(f"FAISS index saved → {OUTPUT_FAISS_TEST}")
    print(f"Total vectors in FAISS: {index.ntotal}")
    print(f"FAISS dimension: {index.d}")
    print("Similarity metric: Inner Product (Cosine compatible)")

    # Validation
    print("\nValidation")
    print("Vector count matches:", index.ntotal == embeddings.shape[0])
    print("Dimension matches:", index.d == embeddings.shape[1])

    return index


# MAIN

if __name__ == "__main__":
    print("\n1. Loading & Preprocessing Documents")
    chunks = load_and_process_documents()
    print(f"Total chunks extracted: {len(chunks)}")

    print("\n2. Embedding Chunks")
    embeddings = generate_embeddings(chunks)

    print("\n3. Building FAISS Index")
    index = build_faiss_index(embeddings)


