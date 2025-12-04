import json
from rag.pipeline import run_rag
from dotenv import load_dotenv

load_dotenv()
TOP_K_DISPLAY = 5


def format_llm_response(parsed: dict) -> str:
    """
    Safe formatter for structured JSON or plain-text fallback.
    If parsed fields are missing, show the raw LLM response.
    
    NOTE: Corrected key mapping to use 'reason' and 'future_steps'
    """
    def normalize(value):
        if value is None:
            return "Not Provided"
        if isinstance(value, list):
            # Check for empty list after stripping potential 'null' strings
            filtered_list = [v for v in value if str(v).lower() not in ("null", "not provided")]
            return "\n    - " + "\n    - ".join(str(v) for v in filtered_list) if filtered_list else "None"
        if isinstance(value, dict):
            return "\n    " + "\n    ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    # Check if structured decision exists (using keys expected by pipeline.py)
    decision = parsed.get("decision")
    reason = parsed.get("reason")
    
    # Check if there is any structured info to display (using 'reason' key)
    if not any([decision, reason]):
        # No structured info, show raw fallback
        return parsed.get("raw") or "LLM returned empty response."

    lines = [
        f"**Eligibility Status:** {normalize(decision)}",
        f"**Confidence Score:** {normalize(parsed.get('confidence'))}",
        # Mapping 'reason' from pipeline.py to 'Reason for Decision' display
        f"**Reason for Decision:** {normalize(reason)}",
        # Mapping 'future_steps' from pipeline.py to 'Actions to Improve' display
        f"**Actions to Improve:** {normalize(parsed.get('future_steps'))}", 
    ]
    return "\n".join(lines)


def main():
    print("SwiftVisa — RAG + Gemini query CLI (Stateless, fallback-safe)\n")

    while True:
        q = input("\nAsk your visa question (or 'exit'): ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        q_clean = " ".join(q.split())

        # --- Run the RAG pipeline ---
        result = run_rag(q_clean, top_k=TOP_K_DISPLAY)
        parsed = result.get("parsed", {})

        # --- Display structured answer ---
        print("\n--- 🗣️ SwiftVisa Assistant ---")
        print(format_llm_response(parsed))

        # --- Display top relevant chunks ---
        print("\n--- 📑 RELEVANT CHUNK UIDs ---")
        retrieved = result.get("retrieved", []) or []
        if retrieved:
            uids = [str(c.get("uid", "No UID")) for c in retrieved[:TOP_K_DISPLAY]]
            print(f"Chunks Used (Top {TOP_K_DISPLAY} UIDs): {', '.join(uids)}")
        else:
            print("No relevant documents were retrieved.")

        # --- Display final blended confidence ---
        print(f"\n**Final blended confidence (retrieval+LLM):** {result.get('final_confidence', 0.0):.3f}")
        print("-----------------------------------")

    print("Goodbye.")


if __name__ == "__main__":
    main()