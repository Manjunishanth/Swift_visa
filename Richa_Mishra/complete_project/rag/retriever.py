# rag/retriever.py
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Data files (match your utils/main outputs)
DATA_DIR = Path("Data")
INDEX_PATH = DATA_DIR / "visa_embeddings.index"
METADATA_JSON = DATA_DIR / "visa_metadata.json"
CHUNKS_JSON = DATA_DIR / "visa_chunks.json"
IDS_NPY = DATA_DIR / "visa_ids.npy"
VECTORS_NPY = DATA_DIR / "visa_embeddings.npy"

class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run main.py to build it.")
        # Load index
        self.index = faiss.read_index(str(index_path))
        # Load metadata & chunks (defensive)
        try:
            with open(METADATA_JSON, "r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
        except Exception:
            self.metadata = {}
        try:
            with open(CHUNKS_JSON, "r", encoding="utf-8") as fh:
                self.chunks = json.load(fh)
        except Exception:
            self.chunks = {}
        # load ids if exist (not strictly necessary)
        try:
            self.ids = np.load(str(IDS_NPY)).astype("int64")
        except Exception:
            self.ids = None

    def retrieve(self, query: str, top_k: int = 5, query_embedding=None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k results. If query_embedding provided, use vector search.
        Otherwise perform naive substring search over stored chunks.
        """
        results = []
        if query_embedding is not None:
            # ensure numpy array of shape (1, d)
            qv = np.array(query_embedding, dtype="float32")
            if qv.ndim == 1:
                qv = qv.reshape(1, -1)
            if qv.ndim != 2:
                raise ValueError(f"query_embedding must be shape (d,) or (1,d); got {qv.shape}")
            # normalize in-place (FAISS expects normalized for IndexFlatIP)
            faiss.normalize_L2(qv)
            scores, ids = self.index.search(qv, top_k)
            for score, rid in zip(scores[0], ids[0]):
                if int(rid) < 0:
                    continue
                sid = str(int(rid))
                meta = self.metadata.get(sid, {}) if isinstance(self.metadata, dict) else {}
                text = self.chunks.get(sid, "") if isinstance(self.chunks, dict) else ""
                results.append({"uid": sid, "score": float(score), "meta": meta, "text": text})
        else:
            # Fallback: substring match
            ql = query.lower()
            for sid, text in (self.chunks.items() if isinstance(self.chunks, dict) else []):
                if ql in (text or "").lower():
                    results.append({"uid": sid, "score": 1.0, "meta": self.metadata.get(sid, {}), "text": text})
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        return results
