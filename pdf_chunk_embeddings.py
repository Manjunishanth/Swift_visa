"""
pdf_chunk_embeddings.py

Reads up to 5 PDF files from an input folder, preprocesses the text, splits it into overlapping character-based chunks,
optionally saves chunks to per-PDF folders, prints each chunk with its character count, and computes embedding vectors.

Embedding options:
 - If OPENAI_API_KEY is present in the environment and `use_openai=True`, the script will call OpenAI embeddings API.
 - Otherwise a local fallback using TfidfVectorizer (scikit-learn) is used as a cheap embedding surrogate (no internet).

Usage:
 python pdf_chunk_embeddings.py --input_dir ./pdfs --output_dir ./chunks_out --chunk_size 1000 --overlap 200 --save_chunks

Dependencies:
 pip install pymupdf numpy scikit-learn openai (openai only if you plan to call OpenAI)

"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np

# Optional imports (may be absent)
try:
    import openai
except Exception:
    openai = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


# -----------------------------
# Utilities
# -----------------------------

def read_pdf_text(path: Path) -> str:
    """Read PDF and return extracted text (all pages concatenated)."""
    doc = fitz.open(str(path))
    parts = []
    for page in doc:
        text = page.get_text("text")
        parts.append(text)
    doc.close()
    return "\n".join(parts)


def preprocess_text(text: str) -> str:
    """Basic preprocessing:
      - fix hyphenation at line breaks (word-\nword -> wordword)
      - normalize whitespace
      - remove non-printable characters
    """
    # fix hyphenation that appears as "word-\nword"
    text = re.sub(r"-\n\s*", "", text)
    # replace any kind of newline with single newline
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse repeated whitespace (but keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # strip leading/trailing
    text = text.strip()
    # remove control chars
    text = re.sub(r"[\x00-\x09\x0B\x0C\x0E-\x1F]+", "", text)
    return text


def chunk_text_charwise(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create overlapping character-based chunks from text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap
    return chunks


# -----------------------------
# Embedding helpers
# -----------------------------

def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if openai is None:
        raise RuntimeError("openai package is not installed")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    openai.api_key = key
    # batch-safe approach
    embeddings = []
    # OpenAI may accept a list directly; for robust code we call per-batch
    resp = openai.Embedding.create(model=model, input=texts)
    for item in resp["data"]:
        embeddings.append(item["embedding"])
    return embeddings


def fallback_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Create a simple TF-IDF vectorization as an offline embedding fallback.
    Returns a numpy array of shape (len(texts), n_features).
    Note: result may be sparse and high-dimensional.
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn not installed; install scikit-learn to use fallback embeddings")
    vec = TfidfVectorizer(max_features=1024, stop_words="english")
    X = vec.fit_transform(texts)
    return X.toarray()


# -----------------------------
# Main processing for one PDF
# -----------------------------

def process_pdf(path: Path, output_dir: Path, chunk_size: int, overlap: int, save_chunks: bool, use_openai: bool):
    print(f"\nProcessing: {path.name}")
    raw = read_pdf_text(path)
    clean = preprocess_text(raw)
    chunks = chunk_text_charwise(clean, chunk_size=chunk_size, overlap=overlap)

    print(f"  Extracted {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

    # Print chunks with char counts
    for i, c in enumerate(chunks, start=1):
        print(f"  Chunk {i}: {len(c)} chars")

    # Optionally save chunks to disk
    out_pdf_dir = output_dir / path.stem
    if save_chunks:
        out_pdf_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(chunks, start=1):
            fname = out_pdf_dir / f"chunk_{i:03d}.txt"
            fname.write_text(c, encoding="utf-8")
        # save metadata
        meta = {"source_pdf": path.name, "n_chunks": len(chunks), "chunk_size": chunk_size, "overlap": overlap}
        (out_pdf_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved chunks to folder: {out_pdf_dir}")

    # Create embeddings
    print("  Creating embeddings...")
    embeddings = None
    try:
        if use_openai:
            embeddings = get_openai_embeddings(chunks)
        else:
            # fallback offline embeddings
            arr = fallback_tfidf_embeddings(chunks)
            embeddings = [list(vec) for vec in arr]
    except Exception as e:
        print("  Warning: failed to create requested embeddings:", str(e))
        print("  Attempting fallback TF-IDF embeddings...")
        arr = fallback_tfidf_embeddings(chunks)
        embeddings = [list(vec) for vec in arr]

    # Save embeddings (as .npy and as json metadata)
    if save_chunks:
        emb_arr = np.array(embeddings, dtype=object)
        # saving as json-friendly structure
        emb_json = {"embeddings": embeddings}
        (out_pdf_dir / "embeddings.json").write_text(json.dumps(emb_json), encoding="utf-8")
        # also save numpy if numeric
        try:
            numeric_arr = np.array(embeddings, dtype=float)
            np.save(out_pdf_dir / "embeddings.npy", numeric_arr)
        except Exception:
            # ignore if embeddings are ragged (e.g., variable-length) or non-numeric
            pass
        print(f"  Saved embeddings to {out_pdf_dir / 'embeddings.json'}")

    return {
        "pdf": path.name,
        "n_chunks": len(chunks),
        "chunks": chunks,
        "embeddings": embeddings,
    }


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./chunks_out", help="Where to save chunks & embeddings")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in characters")
    parser.add_argument("--save_chunks", action="store_true", help="Save chunks and embeddings to disk")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY env var and openai package)")
    parser.add_argument("--max_pdfs", type=int, default=5, help="Maximum number of PDFs to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])[: args.max_pdfs]
    if not pdfs:
        print("No PDF files found in", input_dir)
        sys.exit(1)

    results = []
    for p in pdfs:
        res = process_pdf(p, output_dir, args.chunk_size, args.overlap, args.save_chunks, args.use_openai)
        results.append({"pdf": res["pdf"], "n_chunks": res["n_chunks"]})

    print("\nSummary:")
    for r in results:
        print(f"  {r['pdf']}: {r['n_chunks']} chunks")


if __name__ == "__main__":
    main()
    