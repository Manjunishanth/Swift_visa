import json

# Path to your embeddings.json
EMBEDDINGS_PATH = "chunks_out/USA_Visa_Screening_Details/embeddings.json"

# Load the JSON
with open(EMBEDDINGS_PATH, "r") as f:
    data = json.load(f)

# Print type and keys to understand structure
print("Type of data:", type(data))

if isinstance(data, dict):
    print("Keys in data:", data.keys())
elif isinstance(data, list):
    print("Length of list:", len(data))
    print("Type of first element:", type(data[0]))
    if isinstance(data[0], dict):
        print("Keys in first element:", data[0].keys())
    else:
        print("First element content (truncated):", str(data[0])[:200])
else:
    print(data)
