import os
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Any, Optional
import re

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment (.env).")
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_MODEL = "models/gemini-2.5-flash"


def _extract_text_from_response(resp: Any) -> str:
    """
    Try multiple strategies to extract textual output from the Gemini SDK response.
    Return empty string if nothing found.
    """
    if resp is None:
        return ""

    # 1) direct .text (common)
    try:
        t = getattr(resp, "text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # 2) prompt_feedback / safety - surface it (non-empty)
    try:
        pf = getattr(resp, "prompt_feedback", None)
        if pf:
            # try to get block_reason name if available
            br = getattr(pf, "block_reason", None)
            if br:
                name = getattr(br, "name", str(br))
                return f"ERROR: Gemini blocked output — block_reason: {name}"
            # fallback: stringify prompt_feedback
            return f"ERROR: Gemini prompt_feedback present: {pf}"
    except Exception:
        pass

    # 3) Check candidates (if .text failed)
    try:
        candidates = getattr(resp, "candidates", None) or []
        if isinstance(candidates, (list, tuple)) and candidates:
            texts = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if content and hasattr(content, "parts"):
                    for p in content.parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            texts.append(txt)
            if texts:
                return "\n\n".join(t for t in texts if t).strip()
    except Exception:
        pass
        
    # 4) nothing found
    return ""


def _simplify_prompt_for_retry(original_prompt: str) -> str:
    """
    Return a much simpler prompt that asks for a short factual answer in the expected JSON format.
    Used when the model returns no text (likely safety or format refusal).
    """
    return (
        "Your prior attempt failed. Answer the user's visa question briefly and factually using only the documents provided. "
        "Prioritize a definitive 'eligible' or 'not eligible' decision. Avoid 'partially eligible' unless absolutely necessary. "
        "If information is missing, set values to 'NULL' or an empty array. "
        "Return a JSON object with `eligibility_status`, `reason`, `future_steps`, and `confidence` (0.0 to 1.0). "
        "Do NOT wrap the JSON in markdown code blocks."
    )


def call_gemini(prompt: str,
                model_name: str = DEFAULT_MODEL,
                max_output_tokens: int = 1024, # Reduced from original 4096 to match RAG usage
                temperature: float = 0.0,
                retry_on_empty: bool = True) -> str:
    """
    Call Gemini and return usable text.
    On first failure (empty/blocked), optionally retries with a simplified prompt.
    Always returns a non-empty string (error message if needed).
    """
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY not configured."

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"ERROR: Failed to initialize Gemini model object: {e}"

    def _call_once(p: str) -> (str, Any):
        """Call model.generate_content and return (extracted_text, raw_response)"""
        try:
            resp = model.generate_content(
                p,
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature
                }
            )
        except Exception as e:
            return (f"ERROR: Gemini API call failed: {e}", None)

        text = _extract_text_from_response(resp)
        # Attempt to include prompt_feedback info in message if text is empty
        if not text:
            try:
                pf = getattr(resp, "prompt_feedback", None)
                if pf:
                    br = getattr(pf, "block_reason", None)
                    if br:
                        brname = getattr(br, "name", str(br))
                        text = f"ERROR: Model blocked output — reason: {brname}"
                    else:
                        text = "ERROR: Model returned prompt_feedback without block_reason."
            except Exception:
                pass

        return (text or "", resp)

    # first attempt
    text, raw = _call_once(prompt)
    if text and not text.startswith("ERROR:"):
        return text

    # If empty/blocked and retry enabled, try simplified prompt
    if retry_on_empty:
        try:
            simple = _simplify_prompt_for_retry(prompt)
            text2, raw2 = _call_once(simple)
            if text2 and not text2.startswith("ERROR:"):
                return text2
            # If still empty, return the best error string available
            if text2:
                return text2
        except Exception:
            pass

    # If we reach here, return first non-empty error or a generic message.
    if text:
        return text
    return "ERROR: Model output filtered or empty."