import json
import faiss
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

DATA_DIR = Path("Data")
INDEX_PATH = DATA_DIR / "visa_embeddings.index"
METADATA_JSON = DATA_DIR / "visa_metadata.json"
CHUNKS_JSON = DATA_DIR / "visa_chunks.json"
IDS_NPY = DATA_DIR / "visa_ids.npy"

class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run your build step to create it.")
        self.index = faiss.read_index(str(index_path))

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

        try:
            self.ids = np.load(str(IDS_NPY)).astype("int64")
        except Exception:
            self.ids = None

        # Expanded keywords
        self.financial_keywords = [
            "salary", "income", "annual", "eur", "€", "usd", "rupee", "dollar", "pound",
            "tuition", "fees", "funds", "bank", "savings", "scholarship", "loan", "financial"
        ]
        self.sponsorship_keywords = [
            "sponsor", "sponsorship", "company", "employer", "job offer", "offer", "parents", "guardian", "host"
        ]
        self.study_keywords = [
            "visa", "f1", "f‑1", "study permit", "acceptance letter", "admission", "program", "university", "college", "school"
        ]

    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """Calculates a keyword match score as a ratio of keywords present in the text."""
        if not text or not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        # Use a score capped at 1.0, based on unique keyword hits
        return min(1.0, matches / 3.0) # Cap keyword score based on an arbitrary number of matches (e.g., 3 hits = 1.0)


    def retrieve(self, query: str, top_k: int = 5, query_embedding=None) -> List[Dict[str, Any]]:
        results = []

        # build keyword list from query
        ql = query.lower()
        keywords = [kw for kw in (self.financial_keywords + self.sponsorship_keywords + self.study_keywords) if kw.lower() in ql]

        if query_embedding is not None:
            qv = np.array(query_embedding, dtype="float32")
            if qv.ndim == 1:
                qv = qv.reshape(1, -1)
            # Normalize embedding vector
            faiss.normalize_L2(qv)
            
            # Search FAISS index
            scores, ids = self.index.search(qv, top_k * 3)
            
            # Combine vector search with keyword scoring
            for vector_score, rid in zip(scores[0], ids[0]):
                if int(rid) < 0:
                    continue
                sid = str(int(rid))
                meta = self.metadata.get(sid, {}) if isinstance(self.metadata, dict) else {}
                text = self.chunks.get(sid, "") if isinstance(self.chunks, dict) else ""
                kw_score = self._keyword_match_score(text, keywords) if keywords else 0.0
                
                # Combine scores: 60% vector similarity, 40% keyword match
                combined_score = float(vector_score) * 0.6 + kw_score * 0.4
                
                results.append({
                    "uid": sid,
                    "score": combined_score,
                    "vector_score": float(vector_score),
                    "keyword_score": kw_score,
                    "meta": meta,
                    "text": text
                })
        else:
            # simple keyword matching fallback
            # The original logic for no embedding is preserved but is much slower for large datasets
            for sid, text in (self.chunks.items() if isinstance(self.chunks, dict) else []):
                text_lower = (text or "").lower()
                kw_score = self._keyword_match_score(text, keywords) if keywords else 0.0
                if kw_score > 0:
                    results.append({
                        "uid": sid,
                        "score": kw_score, # Use only keyword score as no vector score is available
                        "vector_score": 0.0,
                        "keyword_score": kw_score,
                        "meta": self.metadata.get(sid, {}),
                        "text": text
                    })

        # Sort by score and return top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Filter to unique UIDs before slicing
        seen_uids = set()
        unique_results = []
        for r in results:
            if r["uid"] not in seen_uids:
                unique_results.append(r)
                seen_uids.add(r["uid"])
        
        results = unique_results[:top_k]

        # Ensure at least top_k chunks (fallback if no good matches)
        if len(results) < top_k and self.chunks:
            chunk_uids = list(self.chunks.keys())
            fallback_uids = [sid for sid in chunk_uids if sid not in seen_uids]
            
            for sid in fallback_uids:
                if len(results) >= top_k:
                    break
                results.append({
                    "uid": sid,
                    "score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                    "meta": self.metadata.get(sid, {}),
                    "text": self.chunks.get(sid, "")
                })
        
        return results