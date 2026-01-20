import streamlit as st
import time
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import answer_question

st.set_page_config(page_title="Visa Analyzer AI", layout="wide")

# ----------------- CSS (Deep Ocean Glass Theme + Animations) -----------------
st.markdown("""
<style>
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* GLOBAL OCEAN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #091c29 0%, #004e92 50%, #2b32b2 100%);
        background-attachment: fixed;
        font-family: 'Outfit', sans-serif;
        color: white;
    }

    /* ANIMATIONS */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(0, 180, 219, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 180, 219, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 180, 219, 0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* MAIN CONTAINER */
    .block-container {
        padding: 3rem 1rem;
        max-width: 900px;
    }
    
    /* GLASS CARD BASE */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 24px;
        padidng: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        animation: slideUp 0.8s ease-out forwards;
    }
    
    /* INPUT AREAS */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a202c !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1.1rem;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a202c !important;
        border-radius: 12px !important;
    }
    
    /* BUTTON */
    .stButton button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 50px !important;
        padding: 0.8rem 3rem !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 210, 255, 0.4) !important;
    }

    /* AI AVATAR & TYPOGRAPHY */
    .ai-avatar {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 10px;
        animation: pulse-glow 2s infinite;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: auto;
        margin-right: auto;
        background: rgba(255,255,255,0.1);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; /* Fallback */
        color: white;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
    }
    
    .result-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 25px;
        border-left: 5px solid #00d2ff;
        margin-top: 20px;
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown('<div style="text-align: center; font-size: 4rem; margin-bottom: 10px;">✈️</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: white;">SwiftVisa</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/2_UserProfile.py", label="👤 User Profile", icon="👤")
    st.page_link("pages/3_Visa_Analyzer.py", label="🤖 Visa Analyzer", icon="🤖")
    
    st.markdown("---")
    st.info("Ready to explore the world? Start by checking your visa requirements!")

# ----------------- UI CONTENT -----------------

st.markdown('<div class="ai-avatar">🤖</div>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center; margin-bottom: 10px;">Visa Analyzer <span style="color: #00d2ff;">AI</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; opacity: 0.8; font-size: 1.2rem; margin-bottom: 40px;">Ask any visa question and get instant, intelligent answers.</p>', unsafe_allow_html=True)

# Container for Inputs
st.markdown('<div class="glass-card" style="padding: 40px;">', unsafe_allow_html=True)

# Country Selector
country_filter = st.selectbox(
    "Choose Visa Country (Context)",
    ["Global / General", "Canada", "USA", "UK", "Germany", "Australia", "UAE", "France"],
    index=0
)

st.markdown("<br>", unsafe_allow_html=True)

# Question Input
user_question = st.text_area(
    "Enter your visa question:",
    placeholder="e.g., Can I work on a tourist visa in Canada?",
    height=100
)

st.markdown("<br>", unsafe_allow_html=True)

# Analyze Button
if st.button("Analyze Query ⚡"):
    if not user_question.strip():
        st.warning("Please enter a question regarding visa policies.")
    else:
        # ----------------- PROGRESS INDICATORS -----------------
        status_container = st.empty()
        
        # Step 1: Embedding
        status_container.info("🧠 Embedding your question...")
        time.sleep(0.8) # Simulate processing
        
        # Step 2: Searching
        status_container.info(f"🔍 Searching {country_filter if country_filter != 'Global / General' else 'global'} visa documents...")
        time.sleep(1.0) # Simulate processing
        
        # Step 3: Generating
        status_container.info("✨ Generating intelligent answer...")
        
        # ----------------- REAL RAG PIPELINE -----------------
        try:
            # Get profile if exists
            user_profile = st.session_state.get("profile", {})
            
            # Combine country context with question if specific country selected
            final_query = user_question
            if country_filter != "Global / General":
                final_query = f"For {country_filter}: {user_question}"

            # Call Backend (imported from src)
            result = answer_question(final_query, user_profile=user_profile)
            
            # success!
            status_container.empty()
            
            # Display Result
            st.markdown(f"""
            <div class="glass-card result-box" style="animation: fadeIn 1s ease;">
                <h3 style="color: #00d2ff; margin-top: 0;">AI Analysis Result</h3>
                <p>{result['answer']}</p>
                <div style="margin-top: 20px; font-size: 0.8rem; opacity: 0.6;">
                    Context used: {len(result['retrieved_chunks'])} document chunks
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            status_container.error(f"An error occurred: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
