# demonstration_inner_product.py
# Small demo: build random normalized vectors, index them with IP, then compare FAISS vs brute force.

import time
import numpy as np
import faiss

def demo_inner_product(num_vectors=5000, dim=384):
    print("\n=== DEMO: FAISS Inner Product vs Brute-force ===")
    print(f"Vectors: {num_vectors}, Dimension: {dim}")

    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).astype("float32")
    # normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
    vectors = vectors / norms

    # build faiss ip index with ids
    ids = np.arange(num_vectors).astype('int64')
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, ids)

    query = vectors[0:1]

    # FAISS search
    t0 = time.time()
    distances, idxs = index.search(query, 3)
    t_faiss = time.time() - t0

    print("\nFAISS results:")
    print("IDs:", idxs[0])
    print("Scores:", distances[0])
    print(f"FAISS time: {t_faiss:.6f} s")

    # Brute-force search using dot product
    t0 = time.time()
    sims = np.dot(vectors, query.T).squeeze()   # shape (num_vectors,)
    top3 = np.argsort(-sims)[:3]
    t_list = time.time() - t0

    print("\nBrute-force results:")
    print("IDs:", top3)
    print("Scores:", sims[top3])
    print(f"Brute-force time: {t_list:.6f} s")

    ratio = t_list / (t_faiss + 1e-9)
    print(f"\nFAISS is ~{ratio:.2f}x faster than brute-force for this run.")

if __name__ == "__main__":
    demo_inner_product()
