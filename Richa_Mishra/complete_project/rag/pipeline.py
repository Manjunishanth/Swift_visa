import json
import re
from typing import List, Dict, Any
import numpy as np
# Dependency for 384-dimension embeddings
from sentence_transformers import SentenceTransformer 

from .retriever import Retriever
from .prompt_builder import build_prompt
from .llm_client import call_gemini
from .logger import log_decision

retriever = Retriever()

# --- Embedding Model Setup (384-dim fix) ---
# NOTE: Ensure 'sentence-transformers' is installed: pip install sentence-transformers
try:
    EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIM = 384
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load Sentence Transformer model. Retrieval will use zero vectors. Error: {e}")
    EMBEDDING_MODEL = None
    EMBEDDING_DIM = 384


def get_embedding(text: str) -> List[float]:
    """
    Generates a 384-dimensional embedding using the Sentence Transformer model.
    """
    if EMBEDDING_MODEL is None:
        return [0.0] * EMBEDDING_DIM 
        
    try:
        embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"ERROR: Sentence Transformer encoding failed: {e}. Using zero vector fallback.")
        return [0.0] * EMBEDDING_DIM


def compute_confidence_from_scores(scores: List[float]) -> float:
    """Maps retrieval scores (cosine similarity) to a 0.0-1.0 confidence range."""
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    arr = np.clip(arr, -1.0, 1.0)
    mapped = (arr + 1.0) / 2.0
    return float(mapped.mean())


def extract_info(text: str) -> Dict[str, Any]:
    """
    Extracts the structured visa decision and metadata from the LLM's raw text output.
    Ensures keys are set to 'reason' and 'future_steps'.
    """
    result = {
        "decision": None,
        "confidence": 0.0,
        "reason": None,          
        "future_steps": [],      
        "raw": text # Preserve raw text for fallback display
    }

    # --- NEW: Specific check for filtered/empty response ---
    if not text or "filtered" in text.lower() or text.startswith("ERROR:"):
        result["reason"] = f"LLM Error/Filter: {text.strip() if text else 'Empty response from model.'}"
        result["decision"] = "ERROR"
        return result
    # --------------------------------------------------------

    # Attempt to parse as strict JSON first
    try:
        cleaned_text = text.strip()
        # Remove JSON markdown wrapper if present
        if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```json"): -len("```")].strip()
        elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```"): -len("```")].strip()
        
        parsed = json.loads(cleaned_text)
        
        if isinstance(parsed, dict):
            # Map LLM output keys to result structure
            result.update({
                "decision": parsed.get("eligibility_status"),
                "reason": parsed.get("reason"),
                "future_steps": parsed.get("future_steps", []), 
                "confidence": parsed.get("confidence", 0.0)
            })
            
            # --- Consistency Cleanup (Same as previous fix) ---
            if result["decision"] is not None: result["decision"] = str(result["decision"]).lower()
            try:
                result["confidence"] = float(result["confidence"])
                result["confidence"] = np.clip(result["confidence"], 0.0, 1.0)
            except (TypeError, ValueError):
                result["confidence"] = 0.0
                
            if not isinstance(result["future_steps"], list): 
                result["future_steps"] = [result["future_steps"]] if result["future_steps"] else []
            
            # Handle "NULL" string values
            null_values = ["null", "not provided", "n/a", "none"]
            if result["decision"] and str(result["decision"]).lower() in null_values: result["decision"] = None
            if result["reason"] and str(result["reason"]).lower() in null_values: result["reason"] = None
            if result["future_steps"] and isinstance(result["future_steps"], list):
                result["future_steps"] = [step for step in result["future_steps"] if str(step).lower() not in null_values]

            return result
            
    except json.JSONDecodeError:
        pass
    except Exception:
        pass

    # --- Fallback (when JSON parsing fails but text exists) ---
    # Use the entire raw text as the reason for manual debugging
    if result["reason"] is None:
        result["reason"] = text.strip()
        
    if result["decision"] is None:
        eligibility_match = re.search(r'(?:eligible|not eligible|partially eligible)', text, re.IGNORECASE)
        if eligibility_match:
            result["decision"] = eligibility_match.group(0).lower()

    return result


def run_rag(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Fully stateless RAG pipeline for SwiftVisa.
    """
    query_embedding = get_embedding(query)
    retrieved = retriever.retrieve(query, top_k=top_k, query_embedding=query_embedding)
    scores = [r.get("score", 0.0) for r in retrieved] if retrieved else []
    
    # Check if retrieval confidence is too low to bother LLM
    retrieval_conf = compute_confidence_from_scores(scores)
    # The current threshold for model output filter is 0.514, keep confidence high
    
    prompt = build_prompt(query, retrieved, max_chars=3000, mode="decision")

    try:
        raw_resp = call_gemini(prompt)
    except Exception as e:
        raw_resp = f"ERROR: LLM call failed outside of client wrapper: {e}"

    parsed = extract_info(raw_resp)

    # Compute blended confidence
    declared_conf = parsed.get("confidence") or 0.0
    try:
        final_conf = 0.6 * retrieval_conf + 0.4 * float(declared_conf)
    except Exception:
        final_conf = retrieval_conf

    # Map yes_no
    dec = (parsed.get("decision") or "").lower()
    yes_no = None
    if "not eligible" in dec:
        yes_no = 0
    elif "eligible" in dec:
        yes_no = 1
    elif "partially" in dec:
        yes_no = 0.5

    answer_obj = {
        "parsed": parsed,
        "final_confidence": float(final_conf),
        "retrieval_score_mean": float(np.mean(scores)) if scores else 0.0,
        "retrieved": retrieved,
        "raw_llm": raw_resp,
        "yes_no": yes_no,
    }

    try:
        log_decision(query, retrieved, answer_obj, raw_prompt=prompt)
    except Exception as e:
        print(f"ERROR: Failed to log decision: {e}")

    return answer_obj