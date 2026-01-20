Visa Screening RAG Pipeline 

This project implements a complete Retrieval Augmented Generation (RAG) system for processing visa screening related PDFs.
It reads documents, cleans them, chunks them, generates embeddings using Hugging Face Transformers, stores the embeddings in a FAISS vector index, initializes a Hugging Face agent, and verifies the entire pipeline with a test query.

Installation
1. Clone or download the repository
git clone <repo>
cd <repo>

2. Install dependencies
pip install -r requirements.txt

Your project requires:
PyPDF2
sentence-transformers
faiss-cpu
transformers
numpy

Project Structure
project/
│── main.py
│── requirements.txt
│── README.md
│
├── data/
│   └── (PDF files)
│
├── index/
│   └── faiss_index.bin
│   └── metadata.pkl
│
└── src/
    ├── preprocess.py
    ├── build_index.py
    └── query_engine.py

How the System Works

This project follows a strict AI pipeline flow used in real world RAG architectures.

Below is the flow your expects:

PROJECT FLOW:
1. Pre processing PDFs

Read PDFs

Extract text

Clean unwanted characters

Normalize formatting

Prepare text for chunking

2. Chunking Documents

Split the cleaned text into 150-word chunks

Maintain 30-word overlap to preserve context

Store each chunk with its document ID

3. Create Embeddings (Hugging Face Transformers)

Use sentence-transformers/all-MiniLM-L6-v2

Convert each chunk → numerical embedding vector

Embeddings capture semantic meaning, not just words

4. Store Embeddings in FAISS

Build FAISS IndexFlatIP vector index

Store all embeddings efficiently

Enables fast similarity search

Used by LLM/RAG systems all over the industry

5. Initialize Hugging Face Agent

Load a summarization model

Ensures NLP capabilities like:

Summaries

Text transformation

Combining top retrieved chunks

6. Test Query

A sample query is embedded

Queried against FAISS

Confirms:

Embeddings work

FAISS search works

Model loads correctly

Pipeline runs end-to-end

7. Final Output

No document chunks printed

Only final confirmation messages such as:

Pre-processing completed
Chunking completed
Embeddings generated
FAISS index created
Hugging Face agent initialized
Query pipeline executed
Pipeline executed successfully
All internship tasks completed

Why These Steps Matter
Pre-processing

Prepares raw PDF text for downstream use.

Chunking

Improves accuracy and reduces noise for semantic search.

Embeddings

Represent text as semantic numerical vectors understood by ML models.

FAISS Indexing

Enables extremely fast retrieval across thousands of embeddings.

Hugging Face Agent

Provides language capabilities such as summarization and transformation.

Final Validation

Confirms that the entire RAG pipeline is functioning properly end-to-end.