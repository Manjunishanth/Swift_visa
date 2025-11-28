# rag/pipeline.py
import json
from typing import Dict, Any, List
from .retriever import Retriever
from .prompt_builder import build_prompt
from .llm_client import call_gemini
from .logger import log_decision
from utils.embedding import get_embedding
from rag.memory import make_profile_key, append_memory
import numpy as np

retriever = Retriever()

def compute_confidence_from_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    arr = np.clip(arr, -1.0, 1.0)
    mapped = (arr + 1.0) / 2.0
    return float(mapped.mean())

def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"raw": text}
    clean = text.strip()
    # remove markdown fences if present
    if clean.startswith("```"):
        clean = clean.strip("` \n")
        if clean.startswith("json"):
            clean = clean[4:].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1:
        return {"raw": clean}
    candidate = clean[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {"raw": clean}

def run_rag(query: str, user_profile: Dict[str, Any] = None, top_k: int = 5):
    # 1) Embed query & retrieve
    query_embedding = get_embedding(query)
    retrieved = retriever.retrieve(query, top_k=top_k, query_embedding=query_embedding)
    scores = [r["score"] for r in retrieved] if retrieved else []

    # 2) Build prompt with memory included
    prompt = build_prompt(query, retrieved, user_profile=user_profile, max_chars=3000, mode="decision")

    # 3) Call LLM
    raw_resp = ""
    try:
        raw_resp = call_gemini(prompt)
    except Exception as e:
        raw_resp = f"ERROR: {e}"

    # 4) Parse JSON safely
    parsed = extract_json(raw_resp)

    # 5) Compute blended confidence
    retrieval_conf = compute_confidence_from_scores(scores)
    declared_conf = parsed.get("confidence")
    if isinstance(declared_conf, (float, int)):
        final_conf = 0.6 * retrieval_conf + 0.4 * float(declared_conf)
    else:
        final_conf = retrieval_conf

    # 6) Map to yes/no for convenience
    decision = parsed.get("decision", None)
    yes_no = None
    if isinstance(decision, str):
        d = decision.lower()
        if "eligible" in d:
            yes_no = "Yes"
        elif "not eligible" in d:
            yes_no = "No"
        elif "insufficient" in d:
            yes_no = "Insufficient information"

    # 7) Save conversation to memory (persistent)
    profile_key = make_profile_key(user_profile or {})
    try:
        append_memory(profile_key, "user", query)
        # store assistant text: prefer JSON-formatted explanation if available
        if isinstance(parsed, dict) and parsed.get("raw") is None:
            # build brief assistant text summary to save
            ass_text = json.dumps({
                "decision": parsed.get("decision"),
                "explanation": parsed.get("explanation"),
                "confidence": parsed.get("confidence"),
                "citations": parsed.get("citations")
            }, ensure_ascii=False)
        else:
            ass_text = parsed.get("raw") if "raw" in parsed else raw_resp
        append_memory(profile_key, "assistant", ass_text)
    except Exception:
        pass

    # 8) Prepare return object
    answer_obj = {
        "parsed": parsed,
        "final_confidence": float(final_conf),
        "retrieval_score_mean": float(np.mean(scores)) if scores else 0.0,
        "retrieved": retrieved,
        "raw_llm": raw_resp,
        "yes_no": yes_no
    }

    # 9) Log decision (to your logger module)
    try:
        log_decision(query, user_profile, retrieved, answer_obj, raw_prompt=prompt)
    except Exception:
        pass

    return answer_obj
