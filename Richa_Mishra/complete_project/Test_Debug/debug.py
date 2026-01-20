import json
import numpy as np
import faiss
import os

print("\n--- Checking FAISS + Metadata + Chunks ---\n")

# 1. Check files exist
files = [
    "visa_embeddings.index",
    "visa_embeddings.npy",
    "visa_ids.npy",
    "visa_metadata.json",
    "visa_chunks.json",
]

for f in files:
    print(f, "exists:", os.path.exists(f))

# 2. Load vector matrix
vectors = np.load("visa_embeddings.npy")
ids = np.load("visa_ids.npy")

print("\nEmbeddings shape:", vectors.shape)
print("IDs shape:", ids.shape)
print("First 5 IDs:", ids[:5])

# 3. Load metadata
with open("visa_metadata.json") as f:
    metadata = json.load(f)

print("\nMetadata entries:", len(metadata))

# 4. Load chunks
with open("visa_chunks.json", encoding="utf-8") as f:

    chunks = json.load(f)

print("Chunks entries:", len(chunks))

# 5. Test FAISS search
index = faiss.read_index("visa_embeddings.index")

query_emb = vectors[0].reshape(1, -1)
scores, retrieved_ids = index.search(query_emb, 3)

print("\nFAISS search result (self-search):")
print("Scores:", scores)
print("IDs:", retrieved_ids)

