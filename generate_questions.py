questions = [
    "What documents are required for a USA tourist visa?",
    "What is Section 221(g) in US visa processing?",
    "Is an interview mandatory for a US B2 visa?",
    "What financial documents are required for US visa?",
    "How long does US tourist visa processing take?",
    "What are common reasons for US visa rejection?",
    "Is medical examination required for US tourist visa?",
    "What is DS-160 form?",
    "What happens at VAC biometric appointment?",
    "Can a US visa be refused after interview?",
]

# Duplicate variations automatically to reach 50
expanded = []
for i in range(5):
    for q in questions:
        expanded.append(f"{q} (case {i+1})")

with open("questions.txt", "w") as f:
    for q in expanded:
        f.write(q + "\n")

print("✅ 50 questions generated")
