import json
import os
from rag.pipeline import run_rag

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
        user_profile = q_data.get("user_profile", {})
        
        print(f"[{i+1}/{len(queries)}] Running query: {query}")
        
        try:
            result = run_rag(query, user_profile=user_profile)
            results.append({"query": query, "user_profile": user_profile, "response": result})
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            results.append({"query": query, "user_profile": user_profile, "response": {"error": str(e)}})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Finished processing queries. Results saved to {output_file}")

if __name__ == "__main__":
    process_queries()
