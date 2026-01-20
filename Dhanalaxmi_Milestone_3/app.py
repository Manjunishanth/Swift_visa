import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ================= PAGE SETUP =================
st.set_page_config(
    page_title="SwiftVisa Eligibility Screening Agent",
    page_icon="🛂",
    layout="centered"
)

# ================= SIDEBAR MENU =================
st.sidebar.title("🌍 SwiftVisa Menu")
menu = st.sidebar.radio("Go to", ["Home", "🛂 Check Eligibility", "🕘 History", "About"])


# ================= PARSE RESPONSE =================
def parse_response(text):
    data = {"eligibility": "UNKNOWN", "confidence": 0.5, "reason": ""}
    for line in text.splitlines():
        if line.startswith("Eligibility:"):
            data["eligibility"] = line.split(":")[1].strip()
        elif line.startswith("Confidence Score:"):
            try:
                data["confidence"] = float(line.split(":")[1].strip())
            except:
                data["confidence"] = 0.5
        elif line.startswith("Reason:"):
            data["reason"] = line.split(":")[1].strip()
    return data

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Home Page ----------------
if menu == "Home":
    st.markdown("""
     <div style="text-align: center;">
        <h1>🌍 Welcome to SwiftVisa!</h1>
        <p style="font-size:18px;">Your AI-powered visa eligibility screening agent.</p>
        <p style="font-size:16px;">✅ Analyze your documents & funds</p>
        <p style="font-size:16px;">📚 Retrieve the latest visa rules</p>
        <p style="font-size:16px;">🤖 Get a clear eligibility decision instantly</p>
        <p style="font-size:16px;">Use the menu on the left to check eligibility or learn more about us.</p>
    </div>
    """, unsafe_allow_html=True)

# =================  ELIGIBILITY =================
elif menu == "🛂 Check Eligibility":
    st.title("🛂 SwiftVisa – Your AI Visa Screening Agent")
    st.caption("Policy-based checks, embassy-style answers ")

    with st.form("visa_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("👤 Full Name")
            age = st.number_input("🎂 Age", 18, 75)
            nationality = st.text_input("🌏 Nationality")

        with col2:
            country = st.selectbox(
                "📍 Destination Country",
                ["United States", "United Kingdom", "Canada", "Schengen", "Ireland"]
            )
            visa_type = st.selectbox(
                "🛂 Visa Type",
                ["Student", "Tourist", "Business", "Work", "Dependent","Family Visit"]
            )

        education = st.text_input("🎓 Education / Job")
        funds = st.text_input("💰 Financial Proof")
        documents = st.text_area("📄 Documents you have")
        question = st.text_area("🗣️ Explain your case")

        submitted = st.form_submit_button("🚀 Check Eligibility")

    if submitted:
        query = f"""
Name: {name}
Age: {age}
Nationality: {nationality}
Destination: {country}
Visa Type: {visa_type}
Education: {education}
Funds: {funds}
Documents: {documents}
Case: {question}
"""

        with st.spinner("🛂 Checking your Eligibilty🤖"):
            from retrieval import retrieve_top_k_documents
            from decision import ask_eligibility_decision
            retrieved = retrieve_top_k_documents(query)
            raw = ask_eligibility_decision(query, retrieved)

        parsed = parse_response(raw)

        st.session_state.history.append({
            "country": country,
            "visa": visa_type,
            "eligibility": parsed["eligibility"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"]   # ✅ NEW
        })

        st.subheader("📌 Visa Decision")

        if parsed["eligibility"] == "YES":
            st.success("🎉 YOUR VISA  IS APPROVED!")
        else:
            st.error("🚫 YOUR VISA IS REJECTED")

        st.metric("🛂 Eligibility", parsed["eligibility"])
        st.metric("📊 Confidence", f"{parsed['confidence']:.2f}")
        st.progress(parsed["confidence"])
        st.info(parsed["reason"])

# =================  HISTORY =================
elif menu == "🕘 History":
    st.title("🕘 Previous Visa Checks")

    if not st.session_state.history:
        st.info("No visa checks yet 💤")
    else:
        for h in reversed(st.session_state.history):
            emoji = "✅" if h["eligibility"] == "YES" else "❌"
            st.write(f"{emoji} {h['country']} | {h['visa']} | Confidence: {h['confidence']:.2f}")
            st.write(f"📝 **Reason:** {h['reason']}")

# ================= ABOUT =================
else:
    st.title("ℹ️ About SwiftVisa")

    st.markdown("""
### 🌍 SwiftVisa – Visa Eligibility Screening Agent

SwiftVisa is an AI-powered visa eligibility screening system designed to help individuals quickly assess their chances of obtaining a visa for different countries. It combines the latest Artificial Intelligence (AI) and Retrieval-Augmented Generation (RAG) techniques to provide personalized, accurate guidance based on user inputs.

Key Features:

AI-Powered Eligibility Check: Uses advanced language models to analyze your personal details, documents, and situation to determine visa eligibility.

RAG-Based Policy Retrieval: Retrieves the most relevant visa policies, rules, and regulations to give informed and up-to-date advice.

Multi-Country Support: Supports popular destinations such as the US, UK, Canada, Schengen and Ireland.
Visa Type Guidance: Checks eligibility for Student, Tourist, Business, Work, Dependent, and Visit Family visas.

User-Friendly Interface: Simple form-based input with clear instructions to guide users.

Secure and Private: No personal data is stored permanently; all inputs are processed safely within the app.

Quick Insights: Get a decision in seconds without waiting for long consultations.


How It Works:

Input Details: Users fill out a form with personal, financial, and document information.

RAG Retrieval: SwiftVisa retrieves relevant visa rules and policy documents.
                
AI Decision: The system analyzes the retrieved data along with user inputs and provides a clear eligibility decision.

Result Display: Users get an instant result on the platform .
                
SwiftVisa is ideal for individuals planning to study, work, travel, or visit family abroad and looking for quick guidance on their visa eligibility without consulting multiple sources
""")


