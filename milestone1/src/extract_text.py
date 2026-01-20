# src/extract_text.py

import os
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path("data/raw_pdfs")
TEXT_DIR = Path("data/texts")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a single PDF file."""
    reader = PdfReader(str(pdf_path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    return "\n".join(text_parts)


def process_all_pdfs(pdf_dir: Path = PDF_DIR, text_dir: Path = TEXT_DIR) -> None:
    """Walk through all PDFs and save corresponding .txt files."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    text_dir.mkdir(parents=True, exist_ok=True)

    for file in pdf_dir.iterdir():
        if file.suffix.lower() != ".pdf":
            continue

        print(f"📄 Processing PDF: {file.name}")
        try:
            text = extract_text_from_pdf(file)
        except Exception as e:
            print(f"❌ Failed to extract {file.name}: {e}")
            continue

        out_path = text_dir / f"{file.stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved text: {out_path}")


if __name__ == "__main__":
    process_all_pdfs()
