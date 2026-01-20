import os
import re
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except:
            text.append("")
    return "\n".join(text)

def clean_text(text):
    text = text.replace("\r", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join([l for l in lines if l])

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def load_and_chunk_pdfs(data_dir="data"):
    print("Pre-processing started")
    all_chunks = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(os.path.join(data_dir, filename))
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            for i, c in enumerate(chunks):
                all_chunks.append({"doc_id": filename, "chunk_id": i, "text": c})
    print("Pre-processing completed")
    print("Chunking completed")
    return all_chunks
