# rag/llm_client.py
"""
Robust Gemini client wrapper.

Requires:
- pip install google-generativeai
- set GEMINI_API_KEY in .env or environment

Returns plain text (string). Caller is responsible for parsing JSON out of text where required.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # We still allow import, but calls will fail with clear error.
    genai_configured = False
else:
    genai.configure(api_key=GEMINI_API_KEY)
    genai_configured = True

DEFAULT_MODEL = "models/gemini-2.5-flash"  # adapt to available model in your account

def list_models():
    if not genai_configured:
        return []
    try:
        models = genai.list_models()
        names = []
        for m in models:
            if isinstance(m, dict):
                names.append(m.get("name"))
            else:
                names.append(getattr(m, "name", None))
        return [n for n in names if n]
    except Exception:
        return []

def call_gemini(prompt: str,
                model_name: str = DEFAULT_MODEL,
                max_output_tokens: int = 512,
                temperature: float = 0.0) -> str:
    """
    Call Gemini and return text. If output is safety-blocked, return helpful error text.
    """
    if not genai_configured:
        raise RuntimeError("GEMINI_API_KEY not configured. Set GEMINI_API_KEY in .env")

    # optional: check availability
    try:
        available = list_models()
        if available and model_name not in available:
            # choose a fallback if possible
            # prefer 'models/gemini-flash-latest' if available
            fallback = None
            for cand in ("models/gemini-flash-latest", "models/gemini-2.5-flash", "models/gemini-pro-latest"):
                if cand in available:
                    fallback = cand
                    break
            if fallback:
                model_name = fallback
    except Exception:
        pass

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature
            }
        )

        # Quick accessor response.text may raise if output filtered.
        try:
            # this returns plain text when available
            text = response.text
            return text
        except Exception:
            # Attempt to extract candidate content manually
            # response.candidates is a list; each candidate may have 'content' parts
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                parts = []
                for cand in candidates:
                    # candidate may be object or dict
                    if isinstance(cand, dict):
                        # dict form (older SDKs)
                        text_part = cand.get("content", "")
                        if not text_part:
                            # try spans
                            text_part = cand.get("output", "")
                    else:
                        # object form: build from cand.output or cand
                        text_part = ""
                        try:
                            # candidate may have "content" or "output" fields
                            text_part = getattr(cand, "content", "") or getattr(cand, "output", "")
                        except Exception:
                            text_part = str(cand)
                    if text_part:
                        parts.append(text_part)
                if parts:
                    return "\n\n".join(parts)

            # Fallback: inspect prompt_feedback / safety ratings for reason
            pf = getattr(response, "prompt_feedback", None)
            msgs = []
            if pf:
                try:
                    if getattr(pf, "block_reason", None):
                        msgs.append(f"Prompt blocked: {getattr(pf.block_reason,'name',str(pf.block_reason))}")
                except Exception:
                    pass
                try:
                    # safety ratings (per-candidate)
                    srs = getattr(pf, "safety_ratings", None) or []
                    for sr in srs:
                        br = getattr(sr, "block_reason", None)
                        if br:
                            msgs.append(f"Output blocked reason: {getattr(br,'name',str(br))}")
                except Exception:
                    pass

            if msgs:
                raise RuntimeError("Model output filtered: " + "; ".join(msgs))
            else:
                raise RuntimeError("Model returned no text parts — output filtered or empty response.")

    except Exception as e:
        # bubble up informative error
        raise RuntimeError(f"Gemini API error: {e}") from e
