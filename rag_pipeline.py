# rag_pipeline.py

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai

# 🔥 LOAD ENV HERE
load_dotenv()

# 🔥 CONFIGURE GEMINI HERE
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(BASE_DIR, "chunks_out", "USA_Visa_Screening_Details")
EMBEDDINGS_PATH = os.path.join(CHUNK_DIR, "embeddings.json")

def get_rag_response(user_query):
    # 1️⃣ Load embeddings
    with open(EMBEDDINGS_PATH, "r") as f:
        data = json.load(f)

    embeddings = np.array(data["embeddings"])

    # 2️⃣ Load chunk texts
    chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.startswith("chunk_")])

    texts = []
    for file in chunk_files:
        with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
            texts.append(f.read())

    # 3️⃣ TF-IDF similarity search
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [user_query])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    best_index = np.argmax(similarities)

    best_chunk = texts[best_index]

    # 4️⃣ Gemini generation
    prompt = f"""
    Answer the question using ONLY the information below.
    

    Context:
    {best_chunk}

    Question:
    {user_query}
    """

    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)

    return response.text
