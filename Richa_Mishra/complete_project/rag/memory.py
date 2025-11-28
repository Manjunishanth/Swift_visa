# rag/memory.py
"""
Persistent memory helper.

Memory file: Data/memory.json

Structure:
{
  "<profile_key>": [
      {"role":"user","text":"...","ts":"2025-11-27T15:00:00"},
      {"role":"assistant","text":"...","ts":"..."}
  ],
  ...
}
"""
import json
from pathlib import Path
from datetime import datetime
import hashlib

MEMORY_PATH = Path("Data") / "memory.json"
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_all():
    if not MEMORY_PATH.exists():
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt file -> rotate and return empty
        backup = MEMORY_PATH.with_suffix(".corrupt.json")
        MEMORY_PATH.rename(backup)
        return {}

def _save_all(data):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_profile_key(profile: dict) -> str:
    """
    Deterministic key for a user profile dict.
    Use sorted json -> sha1 to keep short.
    """
    if not profile:
        base = "{}"
    else:
        base = json.dumps(profile, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def get_memory(profile_key: str, max_items: int = 10):
    data = _load_all()
    convo = data.get(profile_key, [])
    # return last max_items
    return convo[-max_items:]

def append_memory(profile_key: str, role: str, text: str):
    data = _load_all()
    convo = data.get(profile_key, [])
    entry = {"role": role, "text": text, "ts": datetime.utcnow().isoformat()}
    convo.append(entry)
    # keep last 200 entries per profile by default
    convo = convo[-200:]
    data[profile_key] = convo
    _save_all(data)
    return entry

def clear_memory(profile_key: str):
    data = _load_all()
    if profile_key in data:
        data.pop(profile_key)
        _save_all(data)
