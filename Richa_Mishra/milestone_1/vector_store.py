# vector_store.py
import faiss
import numpy as np
import json
from typing import Tuple, Dict, Any, List

def build_faiss_index(embeddings: List[Dict[str, Any]],
                      index_path: str = "visa_embeddings.index",
                      metadata_path: str = "visa_metadata.json",
                      vectors_npy: str = "visa_embeddings.npy",
                      ids_npy: str = "visa_ids.npy",
                      use_inner_product: bool = True
                      ) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], np.ndarray]:
    """
    Build and persist a FAISS index from `embeddings` list.

    Each item in embeddings must be a dict containing:
      - 'unique_id' : int (global unique id)
      - 'embedding' : np.array (1D float32; normalized if using inner product)
      - 'source'    : str (filename)
      - 'chunk_id'  : int (local or global chunk id)

    Returns:
      - vectors: numpy array shape (N, dim)
      - metadata: dict keyed by str(unique_id) -> {"source":..., "chunk_id":...}
      - ids: numpy array of int64 ids in the same order as vectors
    """
    if not embeddings:
        raise ValueError("No embeddings provided to build_faiss_index.")

    # Build id array and vector matrix in the exact same order as the embeddings list
    ids = np.array([int(e['unique_id']) for e in embeddings], dtype='int64')
    vectors = np.vstack([e['embedding'] for e in embeddings]).astype('float32')

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be 2D (N, dim). Got shape: " + str(vectors.shape))

    dim = vectors.shape[1]
    print(f"Building FAISS index with {vectors.shape[0]} vectors of dimension {dim}.")

    # Choose base index type
    if use_inner_product:
        base_index = faiss.IndexFlatIP(dim)
    else:
        base_index = faiss.IndexFlatL2(dim)

    # Use IDMap so FAISS returns your unique IDs
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(vectors, ids)

    # write the index to disk
    faiss.write_index(index, index_path)

    # Save vectors & ids to disk for reliable loading later
    np.save(vectors_npy, vectors)
    np.save(ids_npy, ids)

    # Save metadata as a dict keyed by unique_id (string keys for JSON safety)
    metadata = {str(int(e['unique_id'])): {"source": e['source'], "chunk_id": e['chunk_id']} for e in embeddings}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Stored {vectors.shape[0]} embeddings in FAISS and saved metadata ({metadata_path}).")

    return vectors, metadata, ids


def load_embeddings(index_path: str = "visa_embeddings.index",
                    metadata_path: str = "visa_metadata.json",
                    vectors_npy: str = "visa_embeddings.npy",
                    ids_npy: str = "visa_ids.npy"
                    ) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], np.ndarray]:
    """
    Load persisted embeddings and metadata.

    Returns:
      - vectors: numpy array (N, dim)
      - metadata: dict keyed by string unique_id
      - ids: numpy array of int64 unique ids
    """
    # Prefer loading the saved numpy arrays (robust)
    try:
        vectors = np.load(vectors_npy)
        ids = np.load(ids_npy)
    except Exception as e:
        # As a fallback, try reading index file and reconstructing (less reliable)
        print(f"Warning: could not load numpy arrays ({e}). Attempting to read from FAISS index file.")
        index = faiss.read_index(index_path)
        if not hasattr(index, "ntotal") or index.ntotal == 0:
            raise RuntimeError("FAISS index empty or unreadable.")
        dim = index.d
        ntotal = index.ntotal
        flat = faiss.vector_float_to_array(index.reconstruct_n(0, ntotal))
        vectors = flat.reshape(ntotal, dim)
        # NOTE: reconstructing ids for IndexIDMap is not trivial; prefer saved IDs.
        raise RuntimeError("Could not load ids from FAISS index; please rebuild index with numpy persistence.")

    # load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return vectors, metadata, ids
