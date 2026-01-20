# utils/vector_store.py

import os
import json
import numpy as np
import faiss


# ---------------------------------------------------------
# BUILD FAISS INDEX FROM EMBEDDINGS
# ---------------------------------------------------------
def build_faiss_index(
        embeddings,
        index_path="visa_embeddings.index",
        metadata_path="visa_metadata.json",
        vectors_npy="visa_embeddings.npy",
        ids_npy="visa_ids.npy"
    ):
    """
    Build a FAISS index from SentenceTransformer embeddings.

    Parameters:
        embeddings  -> list of dicts {unique_id, chunk_id, source, embedding}
        index_path  -> where FAISS index will be stored
        metadata_path -> where metadata JSON will be stored
        vectors_npy  -> npy file for raw vectors
        ids_npy      -> npy file for vector IDs

    Returns:
        vectors, metadata, ids
    """

    # Convert list → numpy array
    vectors = np.array([e["embedding"] for e in embeddings]).astype("float32")
    ids = np.array([e["unique_id"] for e in embeddings]).astype("int64")

    dim = vectors.shape[1]
    print(f"🧩 Building FAISS index with {vectors.shape[0]} vectors, dim={dim}")

    # Normalize for cosine similarity (required for IndexFlatIP)
    faiss.normalize_L2(vectors)

    # Build inner-product index
    index = faiss.IndexFlatIP(dim)

    # ID-mapped index (so FAISS returns your chunk IDs)
    id_index = faiss.IndexIDMap(index)
    id_index.add_with_ids(vectors, ids)

    # Save FAISS index
    faiss.write_index(id_index, str(index_path))

    # Save vectors + ids
    np.save(str(vectors_npy), vectors)
    np.save(str(ids_npy), ids)

    # Save metadata
    metadata = {
        str(e["unique_id"]): {
            "source": e["source"],
            "chunk_id": e["chunk_id"]
        }
        for e in embeddings
    }

    with open(str(metadata_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"📦 FAISS index saved → {index_path}")
    print(f"📄 Metadata saved → {metadata_path}")

    return vectors, metadata, ids


# ---------------------------------------------------------
# LOAD INDEX (OPTIONAL)
# ---------------------------------------------------------
def load_faiss_index(index_path):
    """
    Load FAISS index from file.
    """
    return faiss.read_index(str(index_path))
