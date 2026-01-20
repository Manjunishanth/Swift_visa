import os
import json
import numpy as np
import faiss

# === Configuration ===
BASE_PATH = r"C:\Users\Akhil\Downloads\AKHIL MENON BATCH 7 VISA"
EMBEDDING_FOLDER = os.path.join(BASE_PATH, "embeddings")  # Folder with .json embedding files
INDEX_PATH = os.path.join(BASE_PATH, "faiss_index.index")

# === Ensure folder exists ===
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

# === Build FAISS index from .json embeddings ===
def build_faiss_index():
    embedding_files = [
        os.path.join(EMBEDDING_FOLDER, f)
        for f in os.listdir(EMBEDDING_FOLDER)
        if f.endswith(".json")
    ]

    if not embedding_files:
        print("⚠️ No .json embedding files found in 'embeddings'. Please add them and rerun.")
        return

    all_embeddings = []
    for f in embedding_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)

            if isinstance(data, dict) and "embedding" in data:
                emb = np.array(data["embedding"], dtype="float32")
                all_embeddings.append(emb)
                print(f"✅ Loaded: {f} → shape {emb.shape}")
            elif isinstance(data, list) and all("embedding" in item for item in data):
                for item in data:
                    emb = np.array(item["embedding"], dtype="float32")
                    all_embeddings.append(emb)
                print(f"✅ Loaded list: {f} → {len(data)} embeddings")
            else:
                print(f"⚠️ Skipped {f}: unexpected format")
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")

    if not all_embeddings:
        print("⚠️ No valid embeddings found in the JSON files.")
        return

    embeddings_np = np.vstack(all_embeddings)
    print(f"\n📊 Total embeddings: {embeddings_np.shape[0]}")

    dim = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings_np)

    faiss.write_index(faiss_index, INDEX_PATH)
    print(f"💾 FAISS index saved to: {INDEX_PATH}")

# === Run the pipeline ===
if __name__ == "__main__":
    build_faiss_index()