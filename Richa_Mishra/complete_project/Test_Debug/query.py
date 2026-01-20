# (C:\Anaconda\shell\condabin\conda-hook.ps1) ; (conda activate infosys)
# https://aistudio.google.com/api-keys

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer


# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + numpy arrays + metadata dictionary
faiss_index = faiss.read_index("visa_embeddings.index")

embeddings = np.load("visa_embeddings.npy")
ids = np.load("visa_ids.npy")     # contains keys for metadata
with open("visa_metadata.json") as f:
    metadata = json.load(f)       # dict, not list


def search_faiss(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True)

    distances, indices = faiss_index.search(query_emb, top_k)

    print("\nQuery:", query)
    print(f"Top-{top_k} Matches\n")

    for rank, idx in enumerate(indices[0]):
        print("=" * 60)
        print(f"Result #{rank + 1}")
        print(f"FAISS Index : {idx}")
        print(f"Distance    : {distances[0][rank]}\n")

        # Embedding
        print("Embedding Vector:")
        print(embeddings[idx], "\n")

        # Correct metadata access
        chunk_id = str(ids[idx])          # convert to string key
        print("Chunk ID:", chunk_id)

        if chunk_id in metadata:
            print("Metadata:")
            print(metadata[chunk_id])
        else:
            print("Metadata: NOT FOUND")

        print("=" * 60)

    return indices, distances


# Example
if __name__ == "__main__":
    query = "Eligibility criteria for Canadian study visa"
    search_faiss(query, top_k=5)
    print("Embeddings shape:", embeddings.shape)



# - distances: similarity scores (smaller = closer if using L2 distance, larger = closer if using cosine similarity depending on setup).
