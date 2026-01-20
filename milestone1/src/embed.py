# src/embed.py

import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("models/chunks.pkl")
EMBEDDINGS_PATH = Path("models/embeddings.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim vectors


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> List[Dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("rb") as f:
        chunks = pickle.load(f)

    return chunks


def compute_embeddings(
    chunks: List[Dict],
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """
    Returns a numpy array of shape (num_chunks, embedding_dim)
    """
    print(f"🧠 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [ch["text"] for ch in chunks]
    print(f"🔢 Computing embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings.astype("float32")


def build_embeddings(
    chunks_path: Path = CHUNKS_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> None:
    chunks = load_chunks(chunks_path)
    embeddings = compute_embeddings(chunks)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with embeddings_path.open("wb") as f:
        pickle.dump(embeddings, f)

    print(
        f"✅ Saved embeddings: shape={embeddings.shape} → {embeddings_path}"
    )


if __name__ == "__main__":
    build_embeddings()
