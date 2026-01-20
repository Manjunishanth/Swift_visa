# src/rag_pipeline.py

import json
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import google.generativeai as genai

# Paths
CHUNKS_PATH = Path("models/chunks.pkl")
FAISS_INDEX_PATH = Path("models/faiss.index")
RESULTS_PATH = Path("outputs/results.json")
USER_PROFILE_PATH = Path("user_profile.json")

# Embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Load .env
# Load .env
# Search for .env in current directory and parent directories
current_dir = Path(__file__).resolve().parent

# Prioritize root .env (current_dir.parent) over src .env (current_dir)
env_candidates = [
    current_dir.parent / ".env",   # root/.env (First choice)
    current_dir / ".env",          # src/.env (Fallback)
]

env_loaded = False
for env_path in env_candidates:
    if env_path.exists():
        print(f"[INFO] Loading .env from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
        env_loaded = True
        break

if not env_loaded:
    print("[WARNING] No .env file found in src/ or root.")

# Support both variable names, prioritizing GOOGLE_API_KEY (from .env)
# over GEMINI_API_KEY (which might be an expired system env var)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    # Print masked key for security/debugging
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:]
    print(f"DEBUG: Loaded API KEY = {masked_key}")
    genai.configure(api_key=GEMINI_API_KEY)
else:
    # Retrieval will still work; Gemini calls will fail until key is set.
    print("[WARNING] API_KEY (GEMINI_API_KEY or GOOGLE_API_KEY) is not set. Gemini calls will not work.")


# ----- Loading helpers -----


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> List[Dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("rb") as f:
        chunks = pickle.load(f)
    return chunks


def load_faiss_index(
    index_path: Path = FAISS_INDEX_PATH,
) -> faiss.IndexFlatL2:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    return index


def load_user_profile(path: Path = USER_PROFILE_PATH) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----- Retrieval -----

_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[INFO] Loading query embedding model: {EMBED_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model


def embed_query(query: str) -> np.ndarray:
    model = _get_embedding_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vec.astype("float32")


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    Returns a list of chunk dicts with an added 'score' field.
    """
    chunks = load_chunks()
    index = load_faiss_index()

    query_vec = embed_query(question)
    distances, indices = index.search(query_vec, top_k)

    ranked_chunks: List[Dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        ch = dict(chunks[int(idx)])  # copy to avoid mutating original
        ch["score"] = float(dist)
        ranked_chunks.append(ch)

    return ranked_chunks


# ----- Prompt + Gemini -----


def build_prompt(
    question: str,
    chunks: List[Dict],
    user_profile: Optional[Dict] = None,
    strict: bool = True,
) -> str:
    """
    Build a STRICT, PDF-based prompt for Gemini.
    """
    context_blocks = []
    for i, ch in enumerate(chunks, start=1):
        context_blocks.append(
            f"[CHUNK {i} | source: {ch.get('source', 'unknown')}]\n{ch['text']}"
        )

    context_text = "\n\n".join(context_blocks)
    profile_text = json.dumps(user_profile or {}, indent=2)

    if strict:
        prompt = f"""
You are a strictly rule-based VISA POLICY assistant called SwiftVisa.

You can ONLY answer using the information provided in the visa policy chunks below.
If the answer is not clearly present, you MUST reply EXACTLY with:
"No data found in the provided visa policy."

--------------------
USER PROFILE (JSON):
{profile_text}
--------------------

VISA POLICY CONTEXT (PDF chunks):
{context_text}
--------------------

INSTRUCTIONS:
- Answer ONLY from the visa policy context above.
- Do NOT guess or hallucinate.
- If information is not present, reply: "No data found in the provided visa policy."
- Keep your answer clear and structured (use bullet points where helpful).

USER QUESTION:
{question}

FINAL ANSWER:
"""
    else:
        # Fallback prompt: prefer PDF info but allow model to answer from general knowledge
        prompt = f"""
You are a VISA POLICY assistant called SwiftVisa.

Use the visa policy chunks below when they contain the answer. If they do not contain the answer, you MAY use your general knowledge to provide a helpful, accurate response and cite when you are using information beyond the provided documents.

--------------------
USER PROFILE (JSON):
{profile_text}
--------------------

VISA POLICY CONTEXT (PDF chunks):
{context_text}
--------------------

INSTRUCTIONS:
- Prefer answers from the provided visa policy context.
- If the policy does not clearly contain the answer, you MAY answer using general knowledge.
- If unsure, be concise and state uncertainty.
- Keep your answer clear and structured (use bullet points where helpful).

USER QUESTION:
{question}

FINAL ANSWER:
"""
    return prompt.strip()


def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    # response.text usually gives the concatenated text
    return response.text


# ----- Answer + Logging -----


def answer_question(
    question: str,
    user_profile: Optional[Dict] = None,
    top_k: int = 3,
    save_result: bool = True,
) -> Dict:
    """
    Full RAG pipeline: retrieve → prompt → Gemini → build result dict.
    """
    if user_profile is None:
        user_profile = load_user_profile()

    retrieved_chunks = retrieve_relevant_chunks(question, top_k=top_k)

    # If no chunks were retrieved, allow Gemini to answer from general knowledge.
    if not retrieved_chunks:
        prompt = build_prompt(question, retrieved_chunks, user_profile, strict=False)
        answer = call_gemini(prompt)
        fallback_used = True
    else:
        # First try: strict PDF-only answer
        prompt = build_prompt(question, retrieved_chunks, user_profile, strict=True)
        answer = call_gemini(prompt)
        # If strict response indicates no data found, fallback to a permissive prompt
        if isinstance(answer, str) and answer.strip() == "No data found in the provided visa policy.":
            prompt = build_prompt(question, retrieved_chunks, user_profile, strict=False)
            answer = call_gemini(prompt)
            fallback_used = True
        else:
            fallback_used = False

    result = {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "fallback_used": fallback_used,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_profile": user_profile,
    }

    if save_result:
        save_result_to_file(result)

    return result


def save_result_to_file(
    result: Dict,
    results_path: Path = RESULTS_PATH,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    if not isinstance(existing, list):
        existing = [existing]

    existing.append(result)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] Result appended -> {results_path}")


if __name__ == "__main__":
    # Ensure stdout handles unicode (for emojis in answers)
    sys.stdout.reconfigure(encoding='utf-8')
    # Simple CLI for quick manual testing
    q = input("Enter your visa question: ")
    res = answer_question(q)
    print("\n=== ANSWER ===\n")
    print(res["answer"])
