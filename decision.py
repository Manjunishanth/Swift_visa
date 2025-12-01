from groq import Groq
from save import save_to_json

client = Groq(api_key="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")

MODEL = "llama-3.1-8b-instant"

def ask_eligibility_decision(query, retrieved_chunks):
    # Build context from retrieved visa policy chunks
    context = ""
    for chunk_file in retrieved_chunks:
        with open(f"chunks/{chunk_file}", "r", encoding="utf-8") as f:
            context += f.read() + "\n\n"

    prompt = f"""
You are a visa eligibility decision engine.
Your job is to decide YES or NO based ONLY on the policy extracts provided below.

USER INPUT:
{query}

POLICY INFORMATION:
{context}

Return the result in this EXACT format:

Eligibility: YES or NO
Reason: short paragraph
Missing Documents (if any): list or say NONE
Citations: chunk filenames used
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content


from retrieval import retrieve_top_k_documents
\
print("\n✨ SwiftVisa – Visa Eligibility Checker\n")

while True:
    query = input("Enter your profile and visa question:\n> ")

    retrieved = retrieve_top_k_documents(query)
    print("\nRetrieved document chunks:", retrieved)

    result = ask_eligibility_decision(query, retrieved)

    print("\n========== ELIGIBILITY RESULT ==========")
    print(result)
    print("========================================\n")

    # Save result
    save_to_json(query, result)

    again = input("Do you want to check another eligibility? (yes/no): ").strip().lower()
    if again not in ["yes","y"]:
        print("\n👋 Exiting SwiftVisa. Have a great day!")
        break
