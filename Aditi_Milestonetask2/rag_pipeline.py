from src.query_engine import VisaRAG
from src.gemini_client import ask_gemini
import numpy as np

rag = VisaRAG()


def calculate_confidence(scores):
    if not scores:
        return 0.0, "Low"

    avg = round(min(1.0, max(0.0, np.mean(scores))), 2)

    if avg > 0.75:
        label = "High"
    elif avg > 0.45:
        label = "Medium"
    else:
        label = "Low"

    return avg, label


def detect_country(question):
    q = question.lower()
    if "india" in q or "indian" in q:
        return "india"
    if "uk" in q or "united kingdom" in q or "britain" in q:
        return "uk"
    if "ireland" in q:
        return "ireland"
    if "schengen" in q or "europe" in q:
        return "schengen"
    if "us" in q or "usa" in q or "america" in q:
        return "us"
    return None


def terminal_rag_pipeline(question):

    print("\n🔍 Searching relevant country PDFs only...")

   
    country = detect_country(question)


    all_results = rag.query(question, top_k=40)

    if country:
        filtered = [r for r in all_results if country in r["doc_id"].lower()]

        
        results = filtered[:8] if filtered else all_results[:8]
    else:
        results = all_results[:8]

    if not results:
        return "No matching documents found.", 0.0, "Low", []

    context = ""
    scores = []
    sources = []

    for r in results:
        context += f"[{r['doc_id']}] {r['text']}\n"
        scores.append(r['score'])
        sources.append((r["doc_id"], r["chunk_id"], round(r["score"], 3)))

    prompt = f"""
You are a VISA DOCUMENT ANALYST.

Answer using ONLY the visa data below.

--------------------
DOCUMENTS:
{context}
--------------------

QUESTION: {question}

FORMAT answer exactly like:

COUNTRY: <name>

VISA TYPE:
<Type>

ELIGIBILITY:
- short bullets

DOCUMENTS REQUIRED:
- short bullets

NOTES:
- short points

FINAL SUMMARY:
1–2 short lines.

Never mention other countries unless asked.
Keep it SHORT.
Do NOT invent anything.
It Should be related to the documents present.
If not related to the documents write data not present.
"""

    answer = ask_gemini(prompt)

    conf, label = calculate_confidence(scores)

    return answer, conf, label, sources
