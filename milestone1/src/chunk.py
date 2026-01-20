# src/chunk.py

import pickle
from pathlib import Path
from typing import List, Dict

TEXT_DIR = Path("data/texts")
CHUNKS_PATH = Path("models/chunks.pkl")


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Example:
    - chunk_size=700, overlap=100 → step = 600
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks(
    text_dir: Path = TEXT_DIR,
    chunks_path: Path = CHUNKS_PATH,
) -> None:
    """
    Read all .txt files and create a list of chunk dicts:
    {
      "id": int,
      "text": str,
      "source": "filename.txt"
    }
    Saved to models/chunks.pkl
    """
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    chunks_dir = chunks_path.parent
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict] = []
    chunk_id = 0

    for txt_file in text_dir.iterdir():
        if txt_file.suffix.lower() != ".txt":
            continue

        print(f"📝 Chunking: {txt_file.name}")
        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()

        raw_chunks = chunk_text(text)
        for ch in raw_chunks:
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": ch,
                    "source": txt_file.name,
                }
            )
            chunk_id += 1

    with chunks_path.open("wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Saved {len(all_chunks)} chunks → {chunks_path}")


if __name__ == "__main__":
    build_chunks()
