import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================= PATH CONFIG =================
BASE_PATH = r"C:\Users\akhil\Downloads\AKHIL MENON BATCH 7 VISA"
CHUNK_FOLDER = os.path.join(BASE_PATH, "chunked_output")
LANGCHAIN_DB_FOLDER = os.path.join(BASE_PATH, "langchain_faiss_db")

# ================= CHECK FOLDER =================
if not os.path.exists(CHUNK_FOLDER):
    print(" chunked_output folder NOT found.")
    exit()

print(" chunked_output folder found")
print(" Files detected:")

files = os.listdir(CHUNK_FOLDER)
for file_name in files:
    print(" -", file_name)
# ================= LOAD EMBEDDINGS =================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================= LOAD AND CHUNK TXT FILES =================
texts = []
metadatas = []

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

for file_name in files:
    if file_name.lower().endswith(".txt"):
        file_path = os.path.join(CHUNK_FOLDER, file_name)
        with open(file_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read().replace("\n", " ").strip()  # flatten lines
            file_chunks = chunk_text(content)
            for idx, chunk in enumerate(file_chunks):
                if chunk:
                    texts.append(chunk)
                    metadatas.append({
                        "source": file_name,
                        "chunk_id": idx
                    })

print(f"\n Total chunks loaded: {len(texts)}")

if len(texts) == 0:
    print("\n No text chunks found inside TXT files.")
    exit()

# ================= LOAD OR CREATE MAIN FAISS DB =================
if os.path.exists(LANGCHAIN_DB_FOLDER) and os.listdir(LANGCHAIN_DB_FOLDER):
    print("\n Loading existing LangChain FAISS DB...")
    db = FAISS.load_local(
        folder_path=LANGCHAIN_DB_FOLDER,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print("\n Creating new LangChain FAISS DB...")
    db = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    db.save_local(LANGCHAIN_DB_FOLDER)
    print(" DB saved successfully")

# ================= QUERY LOOP =================
country_list = ["canada", "us", "ireland", "schengen", "uk"]

while True:
    query = input("\nEnter query (or type 'exit'): ").strip()
    if query.lower() == "exit":
        print("Program exited.")
        break

    # Detect country from query
    selected_country = None
    for c in country_list:
        if c in query.lower():
            selected_country = c
            break

    # Filter only relevant country chunks
    if selected_country:
        filtered_texts = []
        filtered_metadatas = []
        for text, meta in zip(texts, metadatas):
            if selected_country in meta.get("source", "").lower():
                filtered_texts.append(text)
                filtered_metadatas.append(meta)

        if filtered_texts:
            db_country = FAISS.from_texts(filtered_texts, embedding_model, metadatas=filtered_metadatas)
            results = db_country.similarity_search(query, k=5)
        else:
            print(f" No chunks found for country: {selected_country}")
            continue
    else:
        # If no country detected, search all documents
        results = db.similarity_search(query, k=5)

    # ================= PRINT RESULTS =================
    if not results:
        print(" No results found.")
    else:
        print("\n========== TOP RESULTS ==========")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content)
            print("Source:", doc.metadata.get("source", "Unknown"), "| Chunk ID:", doc.metadata.get("chunk_id", "N/A"))

