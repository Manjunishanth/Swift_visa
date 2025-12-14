import json
import subprocess

results = []

with open("questions.txt") as f:
    questions = f.readlines()

for q in questions:
    q = q.strip()
    print(f"🔎 Asking: {q}")

    process = subprocess.Popen(
        ["python3", "query_llm.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output, error = process.communicate(q + "\n")

    results.append({
        "question": q,
        "response": output
    })

with open("qa_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ 50 Q&A saved to qa_results.json")
