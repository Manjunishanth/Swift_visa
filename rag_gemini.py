import os
import google.generativeai as genai
import json

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- FILE MAPPING ----------------
COUNTRY_FILES = {
    "usa": "us_chunks.txt",
    "united states": "us_chunks.txt",
    "canada": "canada_chunks.txt",
    "uk": "uk_chunks.txt",
    "ireland": "ireland_chunks.txt",
    "schengen": "schengen_chunks.txt"
}

CHUNKS_FOLDER = "chunked_output"

# ---------------- LOAD CHUNKS FROM ONE FILE ----------------
def load_country_chunks(filename):
    path = os.path.join(CHUNKS_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("--- CHUNK")
    clean_chunks = []

    for c in chunks:
        c = c.strip()
        if c:
            clean_chunks.append(c)

    print(f" Loaded {len(clean_chunks)} chunks from {filename}")
    return clean_chunks

# ---------------- FLATTEN JSON CHUNK ----------------
def flatten_json_chunk(chunk_text):
    try:
        # Remove the chunk number prefix if exists
        if chunk_text.startswith("{"):
            data = json.loads(chunk_text)
        else:
            json_start = chunk_text.find("{")
            data = json.loads(chunk_text[json_start:])

        flat_lines = []

        def recursive_flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    recursive_flatten(v, f"{prefix}{k}: ")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    recursive_flatten(item, f"{prefix}- ")
            else:
                flat_lines.append(f"{prefix}{obj}")

        recursive_flatten(data)
        return "\n".join(flat_lines)
    except Exception as e:
        # fallback: return raw text if JSON parse fails
        return chunk_text

# ---------------- USER QUERY ----------------
query = input("\nEnter your eligibility query: ").lower()

selected_file = None
for key in COUNTRY_FILES:
    if key in query:
        selected_file = COUNTRY_FILES[key]
        break

if not selected_file:
    raise ValueError(" Country not detected in query (use USA/Canada/UK/Ireland/Schengen)")

chunks = load_country_chunks(selected_file)

# ---------------- FLATTEN CHUNKS ----------------
flattened_chunks = [flatten_json_chunk(c) for c in chunks]

# ---------------- RETRIEVE TOP CHUNKS ----------------
# For simplicity, feed all flattened chunks to LLM
retrieved_chunks = flattened_chunks[:6]  # top 6 chunks

print("\n Retrieved Chunks (flattened):\n")
for i, c in enumerate(retrieved_chunks, 1):
    print(f"{i}. {c[:300]}...\n")

# ---------------- FINAL PROMPT ----------------
context = "\n\n".join(retrieved_chunks)

final_prompt = f"""
You are a visa assistance AI.

The context below contains UK visa information. Use ONLY this context to answer the user's query.

Context:
{context}

Question:
{query}

If the answer is not in the context, say:
"I don't have enough information from the provided documents."

Answer:
"""

# ---------------- GEMINI CALL ----------------
response = model.generate_content(final_prompt)

print("\n LLM FINAL ANSWER:\n")
print(response.text)
