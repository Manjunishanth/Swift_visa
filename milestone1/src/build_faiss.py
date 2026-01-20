# src/build_faiss.py

import pickle
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_PATH = Path("models/embeddings.pkl")
CHUNKS_PATH = Path("models/chunks.pkl")  # for sanity check
FAISS_INDEX_PATH = Path("models/faiss.index")


def load_embeddings(embeddings_path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    with embeddings_path.open("rb") as f:
        embeddings = pickle.load(f)

    return embeddings


def build_faiss_index(
    embeddings_path: Path = EMBEDDINGS_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
) -> None:
    embeddings = load_embeddings(embeddings_path)

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype="float32")

    num_vectors, dim = embeddings.shape
    print(f"📚 Building FAISS index: {num_vectors} vectors, dim={dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))

    print(f"✅ Saved FAISS index → {faiss_index_path}")


if __name__ == "__main__":
    build_faiss_index()
