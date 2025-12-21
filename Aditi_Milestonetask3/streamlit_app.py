import streamlit as st
import sys
import os
import base64

sys.path.append(os.path.abspath(""))

from src.rag_pipeline import terminal_rag_pipeline


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    bg_image = get_base64_image("data/woo.png")
except FileNotFoundError:
    bg_image = "" 

st.set_page_config(page_title="Swift Visa AI Assistant", layout="wide")

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, .stApp {{
    font-family: 'Poppins', sans-serif;
    
}}

.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
  

}}

/* UNIVERSAL GLASS BUTTON - TEXT ALWAYS BLACK */
.glass-btn button {{
    background: white !important;
    color: white !important;  /* ALWAYS BLACK TEXT */
    padding: 10px 18px !important;
    border-radius: 20px !important;
    border: 1.5px solid rgba(0,0,0,0.25) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    backdrop-filter: blur(6px);
    font-size: 16px;
    width: 140px;
    font-weight: 600;
}}

.glass-btn button:hover {{
    background: white !important;
    color: white !important;  /* HOVER = WHITE TEXT */
}}


/* INPUT CARD SMALL + CLEAN (NO BLACK BAR) */
.input-card {{
    padding: 10px;
    width: 5%;
}}

/* MAKE INPUT BOX SMALLER AND FIX WIDTH */
/* FORCE NARROW INPUT BOX WIDTH */
div.stTextInput > div > div > input {{
    width: 320px !important;    /* change this number as you like */
    max-width: 320px !important;
    min-width: 320px !important;
    padding: 10px !important;
    background: rgba(255,255,255,0.95) !important;
    background: white !important;
    border-radius: 10px !important;
}}

/* Fix full-width input container */
div.stTextInput {{
    width: 330px !important;     /* total container width */
    color:black !important; 
}}

div.stTextInput > div > div > input {{
    width: 330px !important;
    min-width: 330px !important;
    max-width: 330px !important;
    padding: 10px !important;
    background: rgba(255,255,255,0.95) !important;
    color: black !important; /* Input text (user's query) is black */
    border-radius: 10px !important;
}}



/* GENERATE BUTTON (Same Style) */
.gen-btn button {{
    background: white !important;
    color: black !important;  /* ALWAYS BLACK TEXT */
    padding: 12px 24px !important;
    border-radius: 12px !important;
    border: 1.5px solid rgba(0,0,0,0.25) !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    width: 180px !important;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}}

.gen-btn button:hover {{
    background: black !important;
    color: white !important;  /* HOVER = WHITE TEXT */
}}


/* RESULT CARD */
.result-card {{
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 12px;
    width: 55%;
    backdrop-filter: blur(6px);
    color: white !important; /* force white text */
}}

.result-card * {{
    color: white !important;  /* makes EVERYTHING white */
}}




</style>
""", unsafe_allow_html=True)


col_nav, col_main = st.columns([1, 5])

with col_nav:
    st.markdown("<div class='glass-btn'>", unsafe_allow_html=True)
    home_btn = st.button("🏠 Home")
    about_btn = st.button("ℹ️ About Us")
    contact_btn = st.button("📞 Contact Us")
    st.markdown("</div>", unsafe_allow_html=True)


if about_btn:
    with col_main:
        st.title("About Swift Visa AI")
        st.write("Swift Visa AI Assistant is designed to make visa guidance simple, accessible, and stress free for everyone.")
        st.write("Understanding visa rules can be confusing, so we built an intelligent system that analyzes official visa documents")
        st.write("Our goal is to help people get reliable visa information without misinformation.")
        st.write("With AI document based responses, and a smooth user experience, Swift Visa AI Assistant is your trusted partner.")
    st.stop()

if contact_btn:
    with col_main:
        st.title("Contact Us")
        st.write("We’re here to help! Whether you have questions, need clarification, or want to share feedback, feel free to reach out.")
        st.write("📧 Email: support@swiftvisaai.com")
        st.write("📞 Phone: +91 12345678")
    st.stop()


with col_main:

    st.title("Swift Visa AI Assistant")

   
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    st.subheader("Ask Your Visa Related Question ✈️")

    query = st.text_input(
        "",
        key="visa_input",
        placeholder="Type your visa question…",
        label_visibility="visible"
    , help=None)

    
    st.markdown("<div class='gen-btn'>", unsafe_allow_html=True)
    generate = st.button("Generate")
    st.markdown("</div>", unsafe_allow_html=True)


 
    if generate:
        if query.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Analyzing visa documents..."):
               
                try:
                    answer, confidence, label, sources = terminal_rag_pipeline(query)
                except NameError:
                    answer = "Error: RAG pipeline not found."
                    confidence = "N/A"
                    label = "N/A"
                    sources = []


            st.subheader("AI Response")

            if answer.lower().startswith("data not present") or "error" in answer.lower():
               st.error(f"{answer}")
            else:
               st.markdown("<div class='result-card'>", unsafe_allow_html=True)
               st.markdown(f"<p>{answer}</p>", unsafe_allow_html=True)
               st.success(f"Confidence: {confidence} ({label})")
               st.markdown("</div>", unsafe_allow_html=True) 