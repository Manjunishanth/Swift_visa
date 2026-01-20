# main.py

import os
import json
from pathlib import Path

from utils.nltk_setup import ensure_nltk_resources
from utils.pdf_utils import extract_text_from_pdf, read_text_file, clean_text
from utils.chunking import sentence_chunking
from utils.embedding import get_embedding, print_embedding_info
from utils.vector_store import build_faiss_index

ensure_nltk_resources()

# Input folder structure
DATA_DIR = Path("Data")
PDF_DIR = DATA_DIR / "pdfs"


# ---------------------------------------------------------
# PROCESS DOCUMENTS → Extract text → Chunk → Embed
# ---------------------------------------------------------
def process_documents():
    """
    Reads all PDFs/TXT files inside Data/pdfs/
    → extracts clean text
    → splits into sentence chunks
    → embeds with SentenceTransformer
    → returns list of embeddings + saves chunks JSON
    """
    all_embeddings = []
    chunks_json = {}
    uid = 0

    for file in sorted(PDF_DIR.iterdir()):
        if file.suffix.lower() not in {".pdf", ".txt"}:
            continue

        print(f"📄 Processing: {file.name}")

        # Extract text
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file)
        else:
            text = read_text_file(file)

        text = clean_text(text)

        if not text.strip():
            print("⚠️ Empty file, skipping.")
            continue

        # Sentence-level chunking
        chunks = sentence_chunking(text)

        for chunk in chunks:
            emb = get_embedding(chunk)

            # Save chunk text for retrieval
            chunks_json[str(uid)] = chunk

            all_embeddings.append({
                "unique_id": uid,
                "chunk_id": uid,
                "source": file.name,
                "embedding": emb
            })

            uid += 1

    # Save all chunk texts to JSON
    with open(DATA_DIR / "visa_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Total Chunks Saved: {len(chunks_json)}")
    return all_embeddings


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\n===== STEP 1 — PROCESSING DOCUMENTS =====")
    embeddings = process_documents()

    if not embeddings:
        print("❌ No embeddings generated. Make sure PDFs exist in Data/pdfs/")
        exit()

    print("\n===== STEP 2 — BUILDING FAISS INDEX =====")
    vecs, meta, ids = build_faiss_index(
        embeddings,
        index_path=DATA_DIR / "visa_embeddings.index",
        metadata_path=DATA_DIR / "visa_metadata.json",
        vectors_npy=DATA_DIR / "visa_embeddings.npy",
        ids_npy=DATA_DIR / "visa_ids.npy",
    )

    print(f"\n🎉 Stored {vecs.shape[0]} embeddings.")
    print_embedding_info(vecs[0])
