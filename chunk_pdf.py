import os
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Configure Gemini
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY is None:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=GEMINI_KEY)

# ------------------------
# 1. Read PDF
# ------------------------
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ------------------------
# 2. Preprocessing
# ------------------------
def preprocess(text):
    return " ".join(text.split())

# ------------------------
# 3. Chunking
# ------------------------
def chunk_text(text, chunk_size=800):
    chunks = []
    words = text.split()
    current = []

    for word in words:
        current.append(word)
        if len(" ".join(current)) > chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# ------------------------
# 4. Gemini Embeddings
# ------------------------
def generate_embeddings(chunks, output_name):
    save_dir = "embeddings_output"
    os.makedirs(save_dir, exist_ok=True)

    all_embeddings = []

    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx+1}/{len(chunks)} ...")

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk
        )

        emb = response["embedding"]
        all_embeddings.append(emb)

    # Save to file
    out_path = os.path.join(save_dir, f"{output_name}_embeddings.txt")
    with open(out_path, "w") as f:
        for emb in all_embeddings:
            f.write(",".join(map(str, emb)) + "\n")

    print(f"Saved embeddings → {out_path}")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    pdf_name = "Schengen_Visa_Screening_Details.pdf"
    full_path = os.path.join("pdfs", pdf_name)

    print("Processing:", pdf_name)

    raw = read_pdf(full_path)
    clean = preprocess(raw)
    chunks = chunk_text(clean)

    print("Chunks created:", len(chunks))

    generate_embeddings(chunks, pdf_name.replace(".pdf", ""))
