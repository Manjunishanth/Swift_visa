# README -- Milestone 1 (Swift Visa AI -Based Visa Eligibility Screening Agent)

## 👤 Submitted by: *Nirumalla Ushasree*

## 📅 Milestone: 1

## 🗂 Project: Swift Visa AI --Based Visa Eligibility Screening Agent

------------------------------------------------------------------------

# 🚀 Overview

This milestone implements the complete pipeline for:

-   Loading Visa-related PDF/TXT documents\
-   Cleaning and preprocessing the text\
-   Splitting text into overlapping chunks\
-   Generating embeddings using a transformer model\
-   Building a FAISS vector index for semantic search

------------------------------------------------------------------------

# 🛠 Tech Stack

  Component        Library
  ---------------- -----------------------
  PDF Extraction   PyPDF2
  Text Cleaning    re
  Embeddings       sentence-transformers
  Vector Index     FAISS
  Storage          NumPy
  Language         Python 3

------------------------------------------------------------------------

# 🔍 Step-by-Step Pipeline

## 1️⃣ Preprocessing (PDF/Text → Clean Text)

-   Extract text from PDF\
-   Remove unnecessary symbols\
-   Normalize whitespace\
-   Convert everything into clean continuous text

------------------------------------------------------------------------

## 2️⃣ Chunking

-   Chunk size = 300 words\
-   Overlap = 50 words

------------------------------------------------------------------------

## 3️⃣ Embedding Generation

Model used: **all-mpnet-base-v2**

-   Encode chunks\
-   Normalize vectors\
-   Save as embeddings.npy

------------------------------------------------------------------------

## 4️⃣ Building FAISS Index

-   Inner Product (IP) similarity\
-   Save as faiss_index.bin

------------------------------------------------------------------------

# ▶️ How to Run

    python main.py

------------------------------------------------------------------------

# 📦 Output Files

  File              Purpose
  ----------------- --------------------------
  embeddings.npy    Encoded semantic vectors
  faiss_index.bin   FAISS search index

------------------------------------------------------------------------

# 🎯 Milestone 1 Completed

✔ Full pipeline implemented\
✔ All files generated\
✔ Ready for semantic search integration
