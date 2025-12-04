import json
import os
from rag.pipeline import run_rag
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

def process_queries(query_file="user_queries.json", output_file="query_results.json"):
    if not os.path.exists(query_file):
        print(f"Error: Query file '{query_file}' not found.")
        return

    with open(query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    print(f"Processing {len(queries)} queries...")
    for i, q_data in enumerate(queries):
        query = q_data["query"]
        # Retain user_profile data for output logging, but DO NOT pass it to run_rag
        # as the corrected rag/pipeline.py does not accept this argument.
        user_profile = q_data.get("user_profile", {})
        
        print(f"[{i+1}/{len(queries)}] Running query: {query}")
        
        try:
            # CORRECTION: Removed 'user_profile' from the run_rag call
            result = run_rag(query) 
            results.append({"query": query, "user_profile": user_profile, "response": result})
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            results.append({"query": query, "user_profile": user_profile, "response": {"error": str(e)}})

    with open(output_file, "w", encoding="utf-8") as f:
        # Use ensure_ascii=False for proper display of non-ASCII characters
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Finished processing queries. Results saved to {output_file}")

if __name__ == "__main__":
    process_queries()