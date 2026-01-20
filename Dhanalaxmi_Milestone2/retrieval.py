from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os


CHUNKS_FOLDER = "chunks"
INDEX_FILE = "visa_index.faiss"
EMBEDDINGS_FILE = "visa_embeddings.npy" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

def retrieve_top_k_documents(query):
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load FAISS index + embeddings
    index = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    # List chunk text files in predefined order
    chunk_files = sorted([f for f in os.listdir(CHUNKS_FOLDER) if f.endswith(".txt")])

    # Encode query
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    # FAISS similarity search
    _, indices = index.search(q_emb, TOP_K)

    # Return the retrieved chunk filenames
    return [chunk_files[i] for i in indices[0]]


if __name__ == "__main__":
    query = input("Enter visa question:\n> ")

    # Task 1 → retrieve documents
    retrieved = retrieve_top_k_documents(query)
    print("\nRetrieved document chunks:", retrieved)

