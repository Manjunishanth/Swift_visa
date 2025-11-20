# pdf_utils.py
import pdfplumber
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def read_text_file(txt_path):
    """Read plain .txt file and return content."""
    p = Path(txt_path)
    # Use explicit encoding to avoid surprises
    return p.read_text(encoding="utf-8", errors="ignore")

def clean_text(text):
    """Light cleaning while preserving paragraph/newline structure."""
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\r\x0b\x0c]+', '\n', text)
    # Collapse multiple newlines to a maximum of 2 (preserve paragraph boundaries)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize spaces but keep single newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace bullets
    text = text.replace('•', '-')
    return text.strip()

