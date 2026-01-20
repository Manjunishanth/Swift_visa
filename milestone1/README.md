# Milestone 1 — Data ingestion & retrieval prototype

This folder contains the data and scripts used to build a basic retrieval index (FAISS) over visa-related documents and test retrieval for the project's Milestone 1.

## Contents
- `data/` — raw PDFs, extracted texts and text merges used as the document corpus.
- `models/` — generated index file (`faiss.index`) used for retrieval.
- `src/` — scripts used to extract, chunk, embed and build/search the FAISS index.

## Key files
- `data/texts/` — plain text documents (e.g., `Canada_merged.txt`, `US_merged.txt`, etc.).
- `models/faiss.index` — prebuilt FAISS index (binary index file).
- `src/build_faiss.py` — pipeline entry to build or update the FAISS index.
- `src/extract_text.py` — (optional) code to extract text from `data/raw_pdfs/`.
- `src/chunk.py` — split documents into chunks for embedding.
- `src/embed.py` — generate embeddings for chunks.
- `src/test_retrieval.py` — sample retrieval / test script.

## Setup
1. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements
```

2. Inspect `requirements` at the repo root if you need to adjust packages.

## Typical workflow
Run the pipeline in stages (examples assume the working directory is the repo root):

1. (Optional) Extract text from PDFs to `data/texts/`:

```powershell
python src/extract_text.py
```

2. Chunk the texts:

```powershell
python src/chunk.py
```

3. Create embeddings for chunks:

```powershell
python src/embed.py
```

4. Build the FAISS index:

```powershell
python src/build_faiss.py
```

5. Run the retrieval test:

```powershell
python src/test_retrieval.py
```

Adjust script arguments as needed — check the top of each script for CLI options.

## Notes & next steps
- The provided `models/faiss.index` is a starting point — re-generate it if you change the corpus or embedding model.
- Milestone 2 / 3 add a web UI and analysis tools; the `milestone3/app.py` references the same models/data.

If you'd like, I can: regenerate the index, add a short example query script, or document specific script arguments.
