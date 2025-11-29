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
                temperature: float = 0.0,
                retry_with_safety: bool = True) -> str:
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
            if text and text.strip():
                return text
        except Exception:
            pass
        
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
                    # object form: extract from Content object's parts
                    text_part = ""
                    try:
                        content = getattr(cand, "content", None)
                        if content:
                            # content is a Content object with 'parts' attribute
                            if hasattr(content, "parts"):
                                # parts is list of Part objects, extract text from each
                                text_parts = []
                                for part in content.parts:
                                    if hasattr(part, "text"):
                                        text_parts.append(part.text)
                                    else:
                                        text_parts.append(str(part))
                                text_part = "".join(text_parts)
                            else:
                                # fallback: try to get text directly
                                text_part = str(content)
                        else:
                            # try output field
                            text_part = getattr(cand, "output", "") or ""
                    except Exception:
                        text_part = str(cand)
                if text_part:
                    parts.append(text_part)
            if parts:
                result = "\n\n".join(parts)
                if result and result.strip():
                    return result

        # Response was filtered or empty
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

        # Always retry with simplified prompt on first failure
        if retry_with_safety:
            try:
                # Simplify the prompt: remove JSON formatting, IMPORTANT markers, etc.
                simple_prompt = prompt.replace("JSON", "text").replace("IMPORTANT", "").strip()
                # Make it more conversational
                simple_prompt = f"Answer this visa question based on the provided documents. Keep answer brief and factual:\n\n{simple_prompt}"
                return call_gemini(simple_prompt, model_name, max_output_tokens * 2, temperature, retry_with_safety=False)
            except Exception as e:
                if msgs:
                    raise RuntimeError("Model output filtered: " + "; ".join(msgs))
                raise RuntimeError("Model returned no text parts — output filtered or empty response.")
        
        if msgs:
            raise RuntimeError("Model output filtered: " + "; ".join(msgs))
        raise RuntimeError("Model returned no text parts — output filtered or empty response.")

    except RuntimeError:
        raise
    except Exception as e:
        # bubble up informative error
        raise RuntimeError(f"Gemini API error: {e}") from e
