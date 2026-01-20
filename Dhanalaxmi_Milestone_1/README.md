Visa Document Embedding & FAISS Indexing Pipeline

This project processes Visa Eligibility PDFs, cleans and chunks the extracted text, generates vector embeddings using Sentence Transformers, and stores them in a FAISS vector database for efficient semantic search.

The pipeline includes:

1. PDF Text Extraction
2. Text Cleaning & Preprocessing
3. Chunking 
4. Sentence Transformer Embedding Generation
5. FAISS Index Creation
6. Multi-PDF Automated Pipeline Runner

---

# 🚀 **Features**

###  Extract text from multiple PDF documents

###  Clean text and remove noise

###  Convert text into chunks

###  Generate embeddings using `all-MiniLM-L6-v2`

###  Store embeddings in a FAISS vector index

###  Fully automated folder-to-index pipeline

---

# How It Works (Step-by-Step)

### Extract & Clean PDFs

Each PDF is processed using pdfplumber:

* Extract raw text
* Remove extra spaces
* Remove junk/unwanted characters
* Convert to lowercase
* Save as a `_cleaned.txt` file

---

###  Chunk the Cleaned Text

To preserve context for embeddings, text is split into fixed word limits.


All chunks are saved inside the `/chunks` folder.

---

### Generate Embeddings

Each chunk is encoded using:

```
model = SentenceTransformer("all-MiniLM-L6-v2")
```

All chunk embeddings are stored in:

```
visa_embeddings.npy
```

---

### Create FAISS Vector Index

Embeddings are stored in a FAISS L2 distance index, saved as:

```
visa_index.faiss
```

This index is later used to perform semantic search.

---

###  Run Full Pipeline on a Folder

You simply provide a folder containing Visa PDF documents:

```
Data/
   ├── usa.pdf
   ├── uk.pdf
   ├── australia.pdf
```

Each PDF is processed automatically.
All chunks → embeddings → FAISS index created.

---

# 📁 Project Structure

```
project/
│
├── Data/                      # Folder containing raw PDF files
├── cleaned_texts/             # Auto-created cleaned text files
├── chunks/                    # Auto-created text chunks
│
├── visa_embeddings.npy        # Saved embeddings
├── visa_index.faiss           # FAISS vector index
│
└── pipeline.py                # Main script (your provided code)
```

---

# ⚙️ **Installation**

### 1. Clone or download repository

```
git clone <repo>
cd <repo>
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Your project requires:

```
pdfplumber
sentence-transformers
faiss-cpu
numpy
```

---

# Running the Pipeline

Place all your Visa PDFs inside the Data folder.

Then run:

```
python pipeline.py
```

Output will include:
✔ Total chunks created
✔ Embedding generation logs
✔ Final FAISS index path
