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

    def _keyword_match_score(self, text: str, keywords: list) -> float:
        """
        Calculate keyword match score. Returns 0-1 based on how many keywords are found.
        """
        if not text or not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords) if keywords else 0.0

    def retrieve(self, query: str, top_k: int = 5, query_embedding=None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k results using hybrid approach:
        1. Vector similarity (semantic search)
        2. Keyword matching (for explicit mentions of salary, sponsorship, requirements)
        """
        results = []
        
        # Extract keywords for additional matching
        keywords = []
        query_lower = query.lower()
        financial_keywords = ["salary", "income", "40k", "50k", "60k", "annual", "usd", "dollar", "pound", "rupee", "financial", "fund", "savings"]
        sponsorship_keywords = ["sponsor", "sponsorship", "company", "employer", "employment", "work visa", "job offer"]
        
        for kw in financial_keywords:
            if kw in query_lower:
                keywords.append(kw)
        for kw in sponsorship_keywords:
            if kw in query_lower:
                keywords.append(kw)
        
        if query_embedding is not None:
            # ensure numpy array of shape (1, d)
            qv = np.array(query_embedding, dtype="float32")
            if qv.ndim == 1:
                qv = qv.reshape(1, -1)
            if qv.ndim != 2:
                raise ValueError(f"query_embedding must be shape (d,) or (1,d); got {qv.shape}")
            # normalize in-place (FAISS expects normalized for IndexFlatIP)
            faiss.normalize_L2(qv)
            scores, ids = self.index.search(qv, top_k * 2)  # retrieve more to apply keyword filtering
            
            for score, rid in zip(scores[0], ids[0]):
                if int(rid) < 0:
                    continue
                sid = str(int(rid))
                meta = self.metadata.get(sid, {}) if isinstance(self.metadata, dict) else {}
                text = self.chunks.get(sid, "") if isinstance(self.chunks, dict) else ""
                
                # Boost score if keywords match
                kw_score = self._keyword_match_score(text, keywords) if keywords else 0.0
                combined_score = float(score) * 0.7 + kw_score * 0.3  # weight vector + keyword
                
                results.append({
                    "uid": sid, 
                    "score": combined_score,
                    "vector_score": float(score),
                    "keyword_score": kw_score,
                    "meta": meta, 
                    "text": text
                })
        else:
            # Fallback: substring + keyword match
            ql = query.lower()
            for sid, text in (self.chunks.items() if isinstance(self.chunks, dict) else []):
                text_lower = (text or "").lower()
                
                # Check substring match
                substring_match = 1.0 if ql in text_lower else 0.0
                
                # Check keyword match
                kw_score = self._keyword_match_score(text, keywords) if keywords else 0.0
                
                # Combined score
                combined_score = substring_match * 0.5 + kw_score * 0.5
                
                if combined_score > 0:
                    results.append({
                        "uid": sid, 
                        "score": combined_score, 
                        "vector_score": substring_match,
                        "keyword_score": kw_score,
                        "meta": self.metadata.get(sid, {}), 
                        "text": text
                    })
        
        # Sort by combined score and take top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results
