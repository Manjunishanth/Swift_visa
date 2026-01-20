import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from .preprocess import load_and_chunk_pdfs

INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
META_FILE = os.path.join(INDEX_DIR, "metadata.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_faiss_index(chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "model_name": MODEL_NAME}, f)
    print("Embeddings generated")
    print("FAISS index created")

def rebuild_index(data_dir="data"):
    chunks = load_and_chunk_pdfs(data_dir)
    build_faiss_index(chunks)
