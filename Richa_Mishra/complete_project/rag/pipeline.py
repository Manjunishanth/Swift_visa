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
    
    # Try to find and parse JSON first
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = clean[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed.get("decision"):
                return parsed
        except Exception:
            pass
    
    # If no valid JSON, parse text response manually
    result = {
        "raw": clean,
        "decision": None,
        "explanation": "",
        "confidence": 0.5,
        "citations": [],
        "additional_facts_required": []
    }
    
    # Split by decision keywords to find the main decision (last one mentioned is most authoritative)
    import re
    
    # Find ALL decision statements in the text
    decision_pattern = r'(?:decision|assessment)\s*:?\s*([^.\n,]*(?:eligible|need more|insufficient)[^.\n,]*)'
    matches = list(re.finditer(decision_pattern, clean, re.IGNORECASE))
    
    if matches:
        # Use the LAST decision mentioned as it's likely the final answer
        last_decision = matches[-1].group(1).strip()
        lower_decision = last_decision.lower()
        
        if "not eligible" in lower_decision:
            result["decision"] = "Not Eligible"
        elif "eligible" in lower_decision and "not" not in lower_decision[:20]:
            result["decision"] = "Eligible"
        elif "need more" in lower_decision or "insufficient" in lower_decision:
            result["decision"] = "Insufficient information"
    
    # Extract confidence - look for last mentioned confidence value
    conf_matches = list(re.finditer(r'confidence\s*:?\s*([0-9.]+)', clean, re.IGNORECASE))
    if conf_matches:
        try:
            result["confidence"] = float(conf_matches[-1].group(1))
            # Normalize if it's out of range
            if result["confidence"] > 1.0:
                result["confidence"] = result["confidence"] / 100.0
        except:
            result["confidence"] = 0.5
    
    # Extract citations [1], [2], etc. - get all unique ones
    citations = set(re.findall(r'\[(\d+)\]', clean))
    result["citations"] = sorted([int(c) for c in citations])
    
    # Extract explanation - look for lines with "reason" or "explanation" or just take main content
    # Remove decision lines, take meaningful content
    lines = clean.split('\n')
    explanation_lines = []
    
    for i, line in enumerate(lines):
        lower_line = line.lower()
        # Skip decision lines, confidence lines, citation-only lines
        if any(skip in lower_line for skip in ['decision:', 'confidence:', 'reason:', 'document numbers', 'documents used:', 'brief reason']):
            continue
        if line.strip() and len(line.strip()) > 10 and not line.strip().startswith('['):
            explanation_lines.append(line.strip())
    
    # Join first few substantial lines as explanation
    explanation = ' '.join(explanation_lines[:4])
    # Clean up excessive spacing
    explanation = re.sub(r'\s+', ' ', explanation).strip()
    if explanation and not explanation.endswith('.'):
        # Find the last sentence boundary
        last_period = explanation.rfind('.')
        if last_period > 0:
            explanation = explanation[:last_period + 1]
        else:
            explanation = explanation + '.'
    
    result["explanation"] = explanation if explanation else "Please see the documents for detailed analysis."
    
    return result

def run_rag(query: str, user_profile: Dict[str, Any] = None, top_k: int = 5):
    # 1) Embed query & retrieve
    query_embedding = get_embedding(query)
    retrieved = retriever.retrieve(query, top_k=top_k, query_embedding=query_embedding)
    scores = [r["score"] for r in retrieved] if retrieved else []

    # 2) Check if user provided substantive information (salary, sponsorship, etc.)
    # This helps us avoid "insufficient information" when user DID provide facts
    has_user_facts = False
    if user_profile:
        has_user_facts = any(v for k, v in user_profile.items() if v and str(v).strip())
    
    query_lower = query.lower()
    has_financial_info = any(term in query_lower for term in ["salary", "income", "40k", "50k", "60k"])
    has_sponsorship_info = any(term in query_lower for term in ["sponsor", "sponsorship", "company", "employer"])
    has_substantial_query = has_financial_info or has_sponsorship_info or has_user_facts

    # 3) Build prompt with memory included
    prompt = build_prompt(query, retrieved, user_profile=user_profile, max_chars=3000, mode="decision")

    # 4) Call LLM with higher token limit for complete responses
    raw_resp = ""
    try:
        raw_resp = call_gemini(prompt, max_output_tokens=1024)
    except Exception as e:
        raw_resp = f"ERROR: {e}"

    # 5) Parse JSON safely
    parsed = extract_json(raw_resp)

    # 6) Post-process: if user provided substantial facts but LLM said "insufficient information",
    #    try to be more lenient (boost confidence or ask for clarification differently)
    decision_str = str(parsed.get("decision", "")).lower()
    if has_substantial_query and "insufficient" in decision_str:
        # User provided real facts, but LLM wants more info
        # This is OK, but we can note that user DID provide something
        additional = parsed.get("additional_facts_required", []) or []
        # Filter out trivial requests for facts we already have
        filtered_additional = []
        for fact in additional:
            fact_lower = str(fact).lower()
            if has_financial_info and any(x in fact_lower for x in ["salary", "income", "financial"]):
                continue  # skip, user already mentioned
            if has_sponsorship_info and any(x in fact_lower for x in ["sponsor", "employer", "employment"]):
                continue  # skip, user already mentioned
            filtered_additional.append(fact)
        
        if filtered_additional:
            parsed["additional_facts_required"] = filtered_additional
        else:
            # All facts already provided; change decision to tentative eligibility
            parsed["decision"] = "Eligible (based on information provided)"
            parsed["confidence"] = 0.65  # moderate confidence

    # 7) Compute blended confidence
    retrieval_conf = compute_confidence_from_scores(scores)
    declared_conf = parsed.get("confidence")
    if isinstance(declared_conf, (float, int)):
        final_conf = 0.6 * retrieval_conf + 0.4 * float(declared_conf)
    else:
        final_conf = retrieval_conf

    # 8) Map to yes/no for convenience - check "not" prefix FIRST to avoid conflicts
    decision = parsed.get("decision", None)
    yes_no = None
    if isinstance(decision, str):
        d = decision.lower().strip()
        # Check for "Not Eligible" FIRST (most specific)
        if "not eligible" in d or "not eligible" in d:
            yes_no = "No"
        # Then check positive indicators
        elif "eligible" in d or "yes" in d:
            yes_no = "Yes"
        # Check for insufficient info
        elif "insufficient" in d or "need more" in d:
            yes_no = "Need more information"
        # Default fallback
        else:
            yes_no = None

    # 9) Save conversation to memory (persistent)
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

    # 10) Prepare return object
    answer_obj = {
        "parsed": parsed,
        "final_confidence": float(final_conf),
        "retrieval_score_mean": float(np.mean(scores)) if scores else 0.0,
        "retrieved": retrieved,
        "raw_llm": raw_resp,
        "yes_no": yes_no
    }

    # 11) Log decision (to your logger module)
    try:
        log_decision(query, user_profile, retrieved, answer_obj, raw_prompt=prompt)
    except Exception:
        pass

    return answer_obj
