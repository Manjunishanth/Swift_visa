# 🚀 SwiftVisa – Visa Eligibility Screening Agent (RAG Based)

SwiftVisa is an AI-powered visa query system built using Retrieval-Augmented Generation (RAG).
Instead of allowing the LLM to answer freely, the system retrieves visa policy data from documents and generates a response strictly based on those retrieved documents, ensuring accuracy and reducing hallucinations.

### Project Goals

The system is built to achieve two main objectives:

1. Retrieve the top-K most relevant visa document chunks based on the user's query using embeddings and FAISS similarity search.

2. Send the retrieved document chunks along with the user query to an LLM to generate the final eligibility decision or visa answer.

   
## 🎯 Project Workflow Overview

The project works in two phases:

### Phase 1 — Preprocessing (Only run once)

Performed by week_1.py
This script:

* Reads visa PDF files
* Converts pages to cleaned text
* Splits text into chunks
* Generates embeddings using SentenceTransformer
* Builds FAISS index for similarity search
* Saves:

  * Chunks/ folder containing text chunks
  * visa_embeddings.npy
  * visa_index.faiss

### Phase 2 — Visa Question Answering (Main RAG Pipeline)

When the user asks a visa question:

1. The query embedding is generated.
2. FAISS retrieves the top-K relevant chunk files.
3. Retrieved chunk content + user query are passed to an LLM.
4. The LLM generates the structured final response based on retrieved content.
5. The query and response are saved automatically in JSON.


## 📂 Repository Structure

Chunks/                     → Auto-generated visa policy text chunks
visa_data.json              → Stores saved results (JSON)
Visa_Eligibility/           → Source PDFs or supporting docs

week_1.py                   → Preprocessing (PDF → Chunks → Embeddings → FAISS)
retrieval.py                → Retrieves top-K relevant chunks for a query
decision.py                 → Sends query + chunks to LLM and gets final result
save.py                     → Saves query and generated answer to JSON
visa_embeddings.npy         → Generated embeddings from chunks
visa_index.faiss            → FAISS index generated from embeddings


### System Workflow

Step-by-step flow of the project:

1.User enters a visa-related query.

2.The query is converted into an embedding using SentenceTransformer.

3.FAISS similarity search retrieves the top-K most similar visa document chunks.

4.The retrieved chunks and the user query are merged to form an LLM prompt.

5.The LLM generates the final structured response based on the retrieved policy content.

6.The response is automatically stored inside a JSON file for future reference.

7.User can choose to continue asking additional visa questions.

## ▶️ How to Run the Project

### Step 1: Install Dependencies
pip install sentence-transformers faiss-cpu groq

### Step 2— Run preprocessing only once

This step must be done before asking visa questions.

python week_1.py

After execution, you will see:

* Text chunk files saved in Chunks/
* Embeddings saved as visa_embeddings.npy
* Index saved as visa_index.faiss

### Step 3 — Run retrieval.py, save.py
retrieval.py retrieves top k chunks relevant to the query.

save.py saves the query and response generated from llm as json file.

### Step 4 -Add Groq API key
Open decision.py and set:

client = Groq(api_key="YOUR_API_KEY")--> paste your groq API key

### Step 4 — Start the Visa Eligibility Checker

Execute  main loop script (the one that imports retrieval + decision + save)

python decision.py


### Example Terminal Interaction

✨ SwiftVisa – Visa Eligibility Checker

Enter your profile and visa question:
> I want Canada Business visitor visa to explore.I have invitation letter, funds of CAD 9000 and hotl booking.Will they appEnter your profile and visa question:

Retrieved document chunks: ['Canada_cleaned_chunk15.txt', 'Canada_cleaned_chunk16.txt', 'Canada_cleaned_chunk4.txt', 'CanadRetrieved document chunks: ['Canada_cleaned_chunk15.txt', 'Canada_cleaned_chunk16.txt', 'Canada_cleaned_chunk4.txt', 'Canada_cleaned_chunk0.txt']

========== ELIGIBILITY RESULT ==========
Eligibility: YES
Reason: The user has provided an invitation letter, funds of CAD 9000, and hotel booking, which are required for a Canada BReason: The user has provided an invitation letter, funds of CAD 9000, and hotel booking, which are required for a Canada Business visitor visa. The user's intention to explore Canada as a business visitor is also clear.
Missing Documents: NONE
Citations: policy_information, latest_facts_news
========================================

[INFO] Data saved to visa_data.json
Do you want to check another eligibility? (yes/no):

### Saved Results Format

Every time the system answers a question, it is automatically saved in:

visa_data.json


### Example entry:

{
  "What is the fee for Canada visa?": "Visitor visa fee is 100 CAD and biometrics fee is 85 CAD."
}


## ✔ Strengths of the System

* Answers are based strictly on real visa policy documents
* FAISS enables fast and accurate document similarity search
* Modular architecture makes the system scalable
* JSON history supports analytics or future exporting

