# app.py

import streamlit as st
from rag_pipeline import get_rag_response

st.set_page_config(page_title="Visa RAG Assistant", layout="centered")

st.title("🇺🇸 Visa Information Assistant")
st.write("Ask visa-related questions and get accurate answers from official documents.")

user_query = st.text_area("Enter your question:", height=120)

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            answer = get_rag_response(user_query)
            st.success("Answer Generated")
            st.write(answer)
