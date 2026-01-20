import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_FILE = "index/faiss_index.bin"
META_FILE = "index/metadata.pkl"

class HuggingFaceAgent:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        print("Hugging Face agent initialized")

class VisaRAG:
    def __init__(self):
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.model = SentenceTransformer(meta["model_name"])
        self.index = faiss.read_index(INDEX_FILE)
        self.agent = HuggingFaceAgent()

    def run_test(self):
        q_emb = self.model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
        self.index.search(q_emb, 5)
        print("Query pipeline executed")
