import os
import json
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input folder (current directory where .txt files are)
input_folder = "."  # Current directory
output_folder = "embeddings"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Country mapping based on filenames
country_mapping = {
    "visacanada": "canada",
    "visausa": "usa", 
    "visaireland": "ireland",
    "visauk": "uk",
    "visaschengen": "schengen"
}

# Process each .txt file
for filename in os.listdir(input_folder):
    if filename.endswith(".txt") and filename.startswith("visa"):
        print(f"Processing: {filename}")
        
        # Get country name from filename
        country_key = filename.replace(".txt", "").lower()
        country_name = country_mapping.get(country_key, "unknown")
        
        file_path = os.path.join(input_folder, filename)
        
        # Read the entire file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Create embedding for the entire file
        embedding = model.encode(content).tolist()

        # Save embedding for this country
        output_filename = f"{country_name}_embedding.json"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "country": country_name,
                "original_file": filename,
                "embedding_size": len(embedding),
                "embedding": embedding
            }, f, indent=2)

        print(f"✓ Embedding saved: {output_filename}")

print("All country embeddings completed!")