import os
import json

INPUT_DIR = "chunks_out"
OUTPUT_DIR = "embeddings_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith("_embeddings.txt"):
            txt_path = os.path.join(root, file)

            records = []
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Expected format: chunk_name|||[embedding list]
                try:
                    chunk_name, embedding_str = line.split("|||")
                    embedding = json.loads(embedding_str)

                    records.append({
                        "chunk": chunk_name,
                        "embedding": embedding
                    })
                except Exception as e:
                    print("Skipping line:", e)

            json_name = file.replace("_embeddings.txt", ".json")
            out_path = os.path.join(OUTPUT_DIR, json_name)

            with open(out_path, "w") as out:
                json.dump(records, out, indent=2)

            print(f"✅ Created JSON → {out_path}")
