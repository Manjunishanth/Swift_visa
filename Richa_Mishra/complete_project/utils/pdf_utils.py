# utils/pdf_utils.py

import pdfplumber
import re
import logging
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.image").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)
logging.getLogger("pdfminer.converter").setLevel(logging.ERROR)


# ------------------------------------------------------------
# OCR HELPER (Fallback when pdfplumber fails)
# ------------------------------------------------------------
def ocr_pdf(pdf_path):
    """
    OCR the entire PDF using pdf2image + pytesseract.
    Returns extracted OCR text.
    """
    text = ""
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"❌ PDF2Image failed for {pdf_path}: {e}")
        return ""

    print(f"🔍 Running OCR on scanned PDF: {Path(pdf_path).name}")

    for i, img in enumerate(images):
        try:
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        except Exception as e:
            print(f"⚠️ OCR failed on page {i+1}: {e}")

    return text


# ------------------------------------------------------------
# MAIN TEXT EXTRACTOR
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using pdfplumber.
    If text is missing (scanned PDFs), fallback to OCR.
    """
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                # If text extraction fails → fallback to OCR for that page
                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    # OCR only the specific page image
                    try:
                        img = page.to_image(resolution=300)
                        ocr_text = pytesseract.image_to_string(img.original)
                        text += ocr_text + "\n"
                    except:
                        # If page.to_image fails, fallback entire PDF OCR
                        print("⚠️ Page-level OCR fallback failed, performing full OCR...")
                        return ocr_pdf(pdf_path)
    except:
        # If pdfplumber fails entirely
        print(f"⚠️ pdfplumber failed for {pdf_path}, switching to full OCR...")
        return ocr_pdf(pdf_path)

    # If result still empty → use OCR
    if not text.strip():
        print(f"⚠️ No text extracted → full OCR mode: {Path(pdf_path).name}")
        return ocr_pdf(pdf_path)

    return text


# ------------------------------------------------------------
# TEXT FILE READER
# ------------------------------------------------------------
def read_text_file(txt_path):
    """Read .txt file with safe encoding."""
    p = Path(txt_path)
    return p.read_text(encoding="utf-8", errors="ignore")


# ------------------------------------------------------------
# CLEAN TEXT
# ------------------------------------------------------------
def clean_text(text):
    """Light text normalization keeping structure."""
    # Remove unwanted control characters
    text = re.sub(r'[\r\x0b\x0c]+', '\n', text)

    # Preserve paragraphs (max 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize extra spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace bullet points
    text = text.replace("•", "-")
    text = text.replace("", "-")

    return text.strip()
