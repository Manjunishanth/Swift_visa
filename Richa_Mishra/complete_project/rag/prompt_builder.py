# rag/prompt_builder.py
from typing import List, Dict, Any
from rag.memory import get_memory, make_profile_key

def _format_memory(profile_key: str, max_entries: int = 6) -> str:
    mem = get_memory(profile_key, max_items=max_entries)
    if not mem:
        return ""
    lines = []
    for i, e in enumerate(mem, start=1):
        role = e.get("role")
        txt = e.get("text", "").strip()
        ts = e.get("ts", "")[:19].replace("T", " ")
        lines.append(f"[M{i}] {role.upper()} ({ts}): {txt}")
    return "PAST CONVERSATION (most recent first):\n" + "\n".join(lines) + "\n\n"

def build_prompt(query: str,
                 retrieved: List[Dict[str, Any]],
                 user_profile: Dict[str, Any] = None,
                 max_chars: int = 2500,
                 mode: str = "decision") -> str:
    """
    Build prompt including:
      - short user profile
      - recent persistent memory (via profile key)
      - numbered retrieved snippets (context)
      - strict JSON decision prompt (if mode=="decision")
    """
    profile_text = ""
    profile_key = ""
    if user_profile:
        allowed_keys = ("age", "income", "family_status", "nationality")
        lines = [f"{k}: {user_profile[k]}" for k in allowed_keys if k in user_profile and user_profile[k]]
        if lines:
            profile_text = "USER PROFILE:\n" + "\n".join(lines) + "\n\n"
        # make profile_key to fetch memory
        profile_key = make_profile_key(user_profile)

    memory_text = _format_memory(profile_key) if profile_key else ""

    # Build numbered context
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
                piece = piece[:remaining]
                ctx_parts.append((idx, piece))
            break
        ctx_parts.append((idx, piece))
        total += len(piece)

    context_text = "\n---\n".join([f"[{i}]\n{p}" for i, p in ctx_parts]) or "[no context available]"

    if mode == "decision":
        prompt = f"""
You are an expert immigration assistant named SwiftVisa. Use ONLY the provided information below to make a strict eligibility decision.
{memory_text}{profile_text}
CONTEXT (numbered snippets):
{context_text}

QUESTION:
{query}

TASK:
1) Decide: one of "Eligible", "Not Eligible", or "Insufficient information".
2) Provide a concise explanation (2-5 sentences) citing snippet numbers like [1], [2].
3) Provide a confidence score between 0.0 and 1.0.
4) List the snippet numbers used in "citations".
5) If decision is "Insufficient information", provide an array "additional_facts_required" listing exact missing factual items.

IMPORTANT: Output EXACTLY ONE JSON object and NOTHING else. The JSON schema:
{{
  "decision": "...",
  "explanation": "...",
  "confidence": 0.0,
  "citations": [1,2],
  "additional_facts_required": []
}}
"""
    else:
        # info mode
        prompt = f"""
You are SwiftVisa, an expert immigration assistant. Use ONLY the numbered context snippets below and the recent conversation memory (if any) to answer.
{memory_text}{profile_text}
CONTEXT (numbered snippets):
{context_text}

QUESTION:
{query}

TASK:
- Provide a concise, factual answer (3-8 sentences).
- Cite context snippets inline as [n].
- If information is missing, respond: "Insufficient information in the provided documents to answer fully." and list what is missing.
"""
    return prompt.strip()
