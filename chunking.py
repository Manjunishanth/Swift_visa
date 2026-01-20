import os
import json

def chunk_json_files():
    # Use the actual folder name where your JSON files are
    input_folder = "C:\\Users\\akhil\\Downloads\\AKHIL MENON BATCH 7 VISA"
    
    # Create output folder
    output_folder = os.path.join(input_folder, "chunked_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each JSON file
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".json", "_chunks.txt"))
            
            # Read JSON file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert to string and split into chunks
            json_str = json.dumps(data, indent=2)
            chunks = []
            current_chunk = ""
            
            for line in json_str.split('\n'):
                if len(current_chunk + line) > 500:  # ~500 chars per chunk
                    chunks.append(current_chunk)
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Save chunks to file
            with open(output_path, 'w') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"--- CHUNK {i} ---\n")
                    f.write(chunk)
                    f.write("\n\n")
            
            print(f"Created {output_path} with {len(chunks)} chunks")

# Run the function
chunk_json_files()