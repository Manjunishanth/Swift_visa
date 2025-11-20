# main.py
import os
import logging
from pathlib import Path

from nltk_setup import ensure_nltk_resources
from pdf_utils import extract_text_from_pdf, read_text_file, clean_text
from chunking import sentence_chunking
from embedding import get_embedding, print_embedding_info
from vector_store import build_faiss_index, load_embeddings

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ensure_nltk_resources()

def process_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_embeddings = []
    next_unique_id = 0  # GLOBAL unique chunk id

    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".pdf", ".txt"}:
            filename = entry.name

            try:
                if entry.suffix.lower() == ".pdf":
                    raw_text = extract_text_from_pdf(str(entry))
                else:
                    raw_text = read_text_file(entry)

                if not raw_text or not raw_text.strip():
                    print(f"Skipping (no text): {filename}")
                    continue

                cleaned = clean_text(raw_text)
                chunks = sentence_chunking(cleaned)

                for chunk in chunks:
                    emb = get_embedding(chunk)

                    record = {
                        "source": filename,
                        "chunk_id": next_unique_id,   # UNIQUE globally
                        "unique_id": next_unique_id,  # UNIQUE globally
                        "embedding": emb
                    }

                    all_embeddings.append(record)
                    next_unique_id += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return all_embeddings

if __name__ == "__main__":
    data_folder = "Data"
    embeddings = process_folder(data_folder)
    if embeddings:
        # build_faiss_index now returns (vectors, metadata_dict, ids)
        stored_embeddings, metadata, ids = build_faiss_index(
            embeddings,
            index_path="visa_embeddings.index",
            metadata_path="visa_metadata.json",
            use_inner_product=True
        )

        print(f"Stored {stored_embeddings.shape[0]} embeddings in FAISS.....")

        print("\n--- Summary ---")
        print("Total embeddings stored:", stored_embeddings.shape[0])
        print("Embedding dimension:", stored_embeddings.shape[1])
        print("Metadata count:", len(metadata))

        # Print shape
        print("\nVector matrix shape:", stored_embeddings.shape)

        # Print info for the first embedding
        print_embedding_info(stored_embeddings[0])

        print("\nSample metadata entry for first id:")
        # ids[0] is the unique id for the first row
        first_id = int(ids[0])
        print(metadata.get(str(first_id), f"No metadata for id {first_id}"))
    else:
        print("No embeddings created.")
