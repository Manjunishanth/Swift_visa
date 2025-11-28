import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# -------------------------
# LOAD ALL DATA
# -------------------------

print("Loading index and data...")

index = faiss.read_index("visa_embeddings.index")
vectors = np.load("visa_embeddings.npy")
ids = np.load("visa_ids.npy")

with open("visa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("visa_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# SEARCH FUNCTION
# -------------------------

def search_faiss(query, top_k=5):
    print("\nQuery:", query)

    # Embed query
    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    # Search FAISS
    scores, retrieved_ids = index.search(query_emb, top_k)

    print("\nTop results:")
    for rank, (score, chunk_id) in enumerate(zip(scores[0], retrieved_ids[0])):
        print("\n==============================")
        print("Rank:", rank + 1)
        print("FAISS Score:", score)
        print("Chunk ID:", chunk_id)

        # Print the 384-dim embedding vector
        print("\nEmbedding vector (first 10 dims):")
        print(vectors[chunk_id][:10], "...")

        # Print metadata
        print("\nMetadata:")
        print(metadata[str(chunk_id)])

        # Print actual text chunk
        print("\nChunk Text:")
        print(chunks[str(chunk_id)])
        print("==============================\n")



# -------------------------
# RUN A TEST QUERY
# -------------------------

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search_faiss(user_query, top_k=5)
