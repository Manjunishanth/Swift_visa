import streamlit as st
import datetime

st.set_page_config(page_title="User Profile", layout="wide")

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
        max-width: 1200px;
    }
    
    /* GLASS CARD BASE */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        animation: slideUp 0.8s ease-out forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.3);
    }
    
    /* INPUT AREAS */
    .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stDateInput input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a202c !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1rem;
        min-height: 45px;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        color: #1a202c !important;
    }
    
    div[data-baseweb="select"] span {
        color: #1a202c !important;
    }

    .stMarkdown label p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500;
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
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 210, 255, 0.4) !important;
    }

    /* TYPOGRAPHY */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: white;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        text-align: center; 
        opacity: 0.8; 
        font-size: 1.2rem; 
        margin-bottom: 40px;
    }
    
    .card-title {
        font-size: 1.5rem;
        color: #00d2ff; /* Cyan accent matching existing theme */
        font-weight: 600;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
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

# ----------------- PAGE CONTENT -----------------

st.markdown('<h1 class="hero-title">My Application Profile</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Provide your details to get accurate, AI-powered visa assessments</p>', unsafe_allow_html=True)

# Main Form Container
if "profile" not in st.session_state:
    st.session_state.profile = {}

# Common Lists
countries = ["India", "USA", "UK", "Canada", "Germany", "France", "Australia", "UAE", "Singapore", "Japan", "Other"]

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # --- PERSONAL DETAILS ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Personal & Passport Details</div>', unsafe_allow_html=True)
    
    name = st.text_input("Full Name", value=st.session_state.profile.get("name", ""), placeholder="As shown in passport")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.profile.get("age", 25))
    with c2:
        nationality = st.selectbox("Nationality", countries, index=0)
        
    residence = st.selectbox("Current Country of Residence", countries, index=0, help="Where are you currently living?")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Passport Details**")
    expiry_date = st.date_input("Passport Expiry Date", 
                                value=st.session_state.profile.get("expiry_date", datetime.date.today() + datetime.timedelta(days=365*2)),
                                help="Many countries require at least 6 months validity.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # --- TRAVEL & BACKGROUND ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✈️ Trip & Background Info</div>', unsafe_allow_html=True)
    
    purpose = st.selectbox("Purpose of Travel", 
                           ["Tourism", "Business", "Study", "Work", "Medical", "Transit", "Family Visit"],
                           index=0)
    
    t1, t2 = st.columns(2)
    with t1:
        duration = st.selectbox("Intended Stay", ["< 7 days", "7-14 days", "15-30 days", "1-3 months", "3-6 months", "> 6 months"])
    with t2:
        travel_date = st.date_input("Travel Date (Optional)", value=None, min_value=datetime.date.today())

    st.markdown("<hr style='margin: 20px 0; opacity: 0.2; border-color: white;'>", unsafe_allow_html=True)
    
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Student", "Retired", "Unemployed"])
    
    st.write("**Visa History**")
    rejection = st.radio("Any previous visa rejections?", ["No", "Yes"], horizontal=True)
    
    rejection_country = ""
    rejection_reason = ""
    
    if rejection == "Yes":
        st.markdown("""
        <div style="background: rgba(255,0,0,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,100,100,0.3); margin-top: 10px;">
            <small style="color: #ffcccc;">⚠️ Rejection history is critical for accurate assessment.</small>
        </div>
        """, unsafe_allow_html=True)
        rejection_country = st.text_input("Which country rejected you?")
        rejection_reason = st.text_area("Reason for rejection (if known)")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Save Button
st.markdown("<br>", unsafe_allow_html=True)
save_col1, save_col2, save_col3 = st.columns([1, 2, 1])

with save_col2:
    if st.button("Save Profile & Sync", type="primary", use_container_width=True):
        st.session_state.profile = {
            "name": name,
            "age": age,
            "nationality": nationality,
            "residence": residence,
            "expiry_date": expiry_date,
            "purpose": purpose,
            "duration": duration,
            "travel_date": travel_date,
            "employment": employment,
            "rejection": rejection,
            "rejection_country": rejection_country,
            "rejection_reason": rejection_reason
        }
        st.toast("Profile Saved Successfully!", icon="💾")
        st.balloons()
