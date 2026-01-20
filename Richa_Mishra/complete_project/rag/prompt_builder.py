from typing import List, Dict, Any

def build_prompt(query: str,
                 retrieved: List[Dict[str, Any]],
                 max_chars: int = 3000,
                 mode: str = "decision") -> str:
    """
    Stateless prompt builder: JSON-first, fallback-safe.
    The primary goal is to get a definitive decision and a confidence score.
    """

    # -------- DOCUMENT CONTEXT (trim to max_chars) -------- #
    ctx_parts = []
    total = 0
    for i, r in enumerate(retrieved or []):
        idx = i + 1
        meta = r.get("meta", {}) or {}
        uid = r.get("uid", f"uid_{idx}")
        chunk_id = meta.get("chunk_id", uid)
        source = meta.get("source", meta.get("doc_id", "unknown"))

        header = f"[{idx}] Source: {source} | chunk_id: {chunk_id}\n"
        text = (r.get("text") or r.get("content") or "").strip()
        piece = header + text + "\n"

        if total + len(piece) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                # Add only the remaining part of the current chunk's text
                ctx_parts.append((idx, piece[:remaining]))
            break

        ctx_parts.append((idx, piece))
        total += len(piece)

    context_text = "\n---\n".join([p for i, p in ctx_parts]) \
                   if ctx_parts else "No context was retrieved. Use NULL where needed."

    # -------- FINAL PROMPT -------- #
    prompt = f"""
You are SwiftVisa, an expert immigration assistant.
Your job is to decide YES or NO based **ONLY** on the policy extracts provided below.

**DOCUMENT CONTEXT:**
{context_text}

**USER QUESTION:**
{query}

**INSTRUCTIONS:**
1. **Output:** Must be **JSON format only**. Do NOT wrap the JSON in markdown code blocks (e.g., no ```json ```).
2. **Decision:** Must be **"eligible"**, **"not eligible"**, or **"partially eligible"**. Prioritize "eligible" or "not eligible".
3. **Confidence:** Provide a numerical score from **0.0 (low)** to **1.0 (high)** reflecting the certainty based on the provided context.
4. **Source:** Use **ONLY** provided context. Set missing data to "NULL" or [].
5. **Conciseness:** Keep the 'reason' concise (2-4 sentences).

**OUTPUT:**
Return a JSON object with the following mandatory fields:
{{
  "eligibility_status": "eligible / not eligible / partially eligible",
  "reason": "2-4 sentences based strictly on context",
  "future_steps": ["steps based on context or to improve eligibility"],
  "confidence": 0.0
}}
"""
    return prompt.strip()