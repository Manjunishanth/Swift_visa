# rag/logger.py
import os
import json
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "decision_log.jsonl")

def log_decision(query: str, user_profile: dict, retrieved: list, answer_json: dict, raw_prompt: str = None):
    try:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "user_profile": user_profile,
            "retrieved": [{"uid": r.get("uid"), "score": r.get("score"), "meta": r.get("meta")} for r in (retrieved or [])],
            "answer": answer_json,
            "prompt": (raw_prompt[:2000] + "...") if raw_prompt and len(raw_prompt) > 2000 else raw_prompt
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Do not crash the flow if logging fails
        print(f"[logger] Failed to write log: {e}")
