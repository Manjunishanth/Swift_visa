import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DECISION_LOG_FILE = LOG_DIR / "decision_log.jsonl"
CONVERSATION_LOG_FILE = LOG_DIR / "conversation_log.jsonl"


def log_decision(query: str, retrieved: list, answer_json: dict, raw_prompt: str = None):
    """Log individual visa eligibility decisions"""
    try:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            # Removed user_profile as it was not passed to run_rag
            "retrieved_count": len(retrieved or []),
            "retrieved": [
                {"uid": r.get("uid"), "score": r.get("score"), "source": r.get("meta", {}).get("source", "unknown")}
                for r in (retrieved or [])
            ],
            "decision": answer_json.get("parsed", {}).get("decision"),
            "confidence": answer_json.get("parsed", {}).get("confidence"),
            "final_confidence": answer_json.get("final_confidence"),
            "raw_prompt": raw_prompt is not None
        }
        with open(DECISION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[logger] Failed to write decision log: {e}")


def log_conversation(profile_key: str, role: str, text: str, metadata: dict = None):
    """Log each conversation turn (user query or assistant response)"""
    try:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "profile_key": profile_key,
            "role": role,
            "text": text[:500] if len(text) > 500 else text,
            "metadata": metadata or {}
        }
        with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[logger] Failed to write conversation log: {e}")