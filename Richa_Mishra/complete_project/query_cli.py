# query_cli.py
"""
Interactive CLI with persistent memory support.

Usage:
  python query_cli.py

It will store conversation history per user profile in Data/memory.json
"""
import json
from rag.pipeline import run_rag
from rag.memory import make_profile_key, get_memory, append_memory
from dotenv import load_dotenv

load_dotenv()
TOP_K_DISPLAY = 5

def ask_profile():
    print("Enter user profile fields (press Enter to skip):")
    prof = {}
    age = input("Age: ").strip()
    if age:
        prof["age"] = age
    income = input("Income (currency/year): ").strip()
    if income:
        prof["income"] = income
    family = input("Family status (single/married/with children): ").strip()
    if family:
        prof["family_status"] = family
    nationality = input("Nationality: ").strip()
    if nationality:
        prof["nationality"] = nationality
    return prof

def format_llm_response(parsed: dict, yes_no: str = None) -> str:
    # Extract fields from parsed response
    eligibility_status = parsed.get("decision", "No decision") # 'decision' now maps to eligibility_status
    confidence_score = parsed.get("confidence", None)
    explanation = parsed.get("explanation", "")
    top_relevant_chunks = parsed.get("top_relevant_chunks", [])
    suggestions = parsed.get("suggestions", "")
    citations = parsed.get("citations", []) or [] # general citations, distinct from top_relevant_chunks
    additional = parsed.get("additional_facts_required", []) or []
    raw = parsed.get("raw", "")

    out = "**🤖 SwiftVisa Assistant:**\n\n"
    
    # Display new structured fields
    if eligibility_status:
        out += f"**Eligibility Status:** {eligibility_status}\n"
    if confidence_score is not None:
        out += f"**Confidence Score:** {confidence_score:.2f}\n"
    
    if explanation:
        out += f"**Explanation of Decision:** {explanation}\n\n"
    elif raw:
        out += f"**Analysis:** {raw}\n\n"
    
    if top_relevant_chunks:
        out += "**Top 5 Most Relevant Chunks:** " + ", ".join(f"[{c}]" for c in top_relevant_chunks) + "\n\n"
    
    if suggestions:
        out += f"**Suggestions for Eligibility & Future Steps:**\n{suggestions}\n\n"
    
    # Keep general citations and additional info needed, if any
    if citations and not top_relevant_chunks: # Only show general citations if top 5 not present
        out += "**Documents Used (General):** " + ", ".join(f"[{c}]" for c in citations) + "\n\n"
    if additional:
        out += "**Additional Information Needed:**\n"
        for item in additional:
            out += f"• {item}\n"
    
    # Remove the old yes_no display as it's now covered by Eligibility Status
    # if yes_no:
    #     out += f"**Yes/No:** {yes_no}\n"
    
    return out.strip()

def main():
    print("SwiftVisa — RAG + Gemini query CLI\n")
    profile = ask_profile()
    profile_key = make_profile_key(profile)
    print("\n✓ Profile created. Your conversation history will be remembered in this session.\n")

    while True:
        q = input("\nAsk your visa question (or 'exit'): ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break

        result = run_rag(q, user_profile=profile, top_k=TOP_K_DISPLAY)
        parsed = result.get("parsed", {})
        yes_no = result.get("yes_no")

        # 1) Conversational output
        print("\n--- 🗣️ SwiftVisa Assistant ---")
        print(format_llm_response(parsed, yes_no=yes_no))

        # 2) Show top retrieved chunk UIDs
        print("\n--- 📑 RELEVANT CHUNK UIDs ---")
        retrieved = result.get("retrieved", [])
        if retrieved:
            uids = [str(c.get("uid", "No UID")) for c in retrieved[:TOP_K_DISPLAY]]
            print(f"Chunks Used (Top {TOP_K_DISPLAY} UIDs): {', '.join(uids)}")
        else:
            print("No relevant documents were retrieved.")

        print(f"\n**Final blended confidence (retrieval+LLM):** {result.get('final_confidence', 0.0):.3f}")
        print("-----------------------------------")

    print("Goodbye.")

if __name__ == "__main__":
    main()
