import os
import re
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def preprocess_pdf(pdf_path, output_dir="cleaned_texts"):
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Extracting from: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Cleaning text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:?!()/%\-\n ]', '', text)
    text = text.lower().strip()
    print(f"[DEBUG] Extracted text length: {len(text)} characters")


    file_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_cleaned.txt"
    cleaned_path = os.path.join(output_dir, file_name)

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] Cleaned saved: {cleaned_path}")
    return cleaned_path



# CHUNKER (Split cleaned text into small chunks)

def chunk_text(cleaned_file, max_length=800, output_dir="chunks"):

    os.makedirs(output_dir, exist_ok=True)

    with open(cleaned_file, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = []

    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_length, n)

        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # move to next chunk with overlap
        start = end - overlap
        if start < 0:
            start = 0

    # Save chunks to files
    base = os.path.splitext(os.path.basename(cleaned_file))[0]
    chunk_files = []

    for i, c in enumerate(chunks):
        path = os.path.join(output_dir, f"{base}_chunk{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(c)
        chunk_files.append(path)

    print(f"[INFO] {len(chunk_files)} chunks created for {cleaned_file}")
    return chunk_files

#  EMBEDDING CREATOR

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunk_files):

    embeddings = []
    texts = []

    print("\n===== GENERATING EMBEDDINGS =====")

    for file in chunk_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        texts.append(text)

        # Generate embedding for current chunk
        embedding = model.encode(text)
    
        embedding_np = np.array(embedding)

        print(f"\n[EMBEDDING GENERATED] File: {file}")
        print(f"Shape: {embedding_np.shape}")

        # Append ONLY the current embedding
        embeddings.append(embedding_np)

    # Convert full list to array
    all_embeddings_np = np.array(embeddings)

    print("\n===== FINAL EMBEDDINGS SHAPE =====")
    print(all_embeddings_np.shape)

    # Save once
    np.save("visa_embeddings.npy", all_embeddings_np)
    print("[INFO] Saved all embeddings to visa_embeddings.npy")

    return all_embeddings_np, texts


# VECTOR DATABASE (FAISS)

def store_faiss(embeddings, index_path="visa_index.faiss"):
    """
    Saves embeddings to a FAISS vector index.
    """
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)

    print(f"[INFO] FAISS index created at: {index_path}")
    return index_path



#  PIPELINE RUNNER FOR MULTIPLE PDFs

def run_pipeline_for_folder(pdf_folder):

    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder)
                 if f.lower().endswith(".pdf")]

    
    all_chunk_files = []
    for pdf in pdf_files:
        print(f"\n--- Processing: {pdf} ---")
        cleaned = preprocess_pdf(pdf)
        chunks = chunk_text(cleaned)
        all_chunk_files.extend(chunks)

    print(f"\n[INFO] Total chunks from all PDFs: {len(all_chunk_files)}")
    print(f"[DEBUG] Calling embed_chunks() with {len(all_chunk_files)} chunk files")
    embeddings, chunk_texts = embed_chunks(all_chunk_files)
    print("[DEBUG] Returned from embed_chunks()")


    #embeddings, chunk_texts = embed_chunks(all_chunk_files)
    index_file = store_faiss(embeddings)
    print("\n===== PIPELINE COMPLETED FOR ALL PDFs =====")
    return index_file, chunk_texts




if __name__ == "__main__":
    folder = "Data"   # folder containing  visa PDFs

    run_pipeline_for_folder(folder)

 




