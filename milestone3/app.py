import streamlit as st
import os
import pandas as pd
import plotly.express as px

# CRITICAL: Clear Streamlit cache
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title="SwiftVisa - Your Travel Companion",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CSS (Deep Ocean Glass Theme + Animations) -----------------
st.markdown("""
<style>
    /* Hide Streamlit branding */
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
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
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
    
    /* TYPOGRAPHY */
    .hero-section {
        text-align: center;
        padding: 60px 20px;
        animation: float 6s ease-in-out infinite;
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#eee, #333); /* Fallback */
        background: linear-gradient(to right, #ffffff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
        margin: 0;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 15px;
        font-weight: 400;
    }

    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
    }
    
    .section-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        margin-bottom: 30px;
    }

    /* BUTTONS */
    .cta-button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 15px 40px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 800;
        text-decoration: none;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        display: inline-block;
        margin-top: 30px;
        border: none;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 210, 255, 0.4);
        color: white;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 28, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] h1 {
        color: white;
    }
    
    .stPageLink a {
        background-color: transparent;
        padding: 10px;
        border-radius: 10px;
        transition: background 0.2s;
    }
    
    .stPageLink a:hover {
        background-color: rgba(255,255,255,0.1);
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


# ----------------- HERO SECTION -----------------
st.markdown("""
<div class="hero-section">
    <div class="float">
        <h1 class="hero-title">SwiftVisa</h1>
        <p class="hero-subtitle">Get instant visa answers powered by AI</p>
    </div>
    <a href="/Visa_Analyzer" class="cta-button" target="_self">Start Analysis</a>
</div>
""", unsafe_allow_html=True)

# Spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------- MAP SECTION -----------------
st.markdown('<div id="map-section"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <h2 class="section-title">🌍 Explore Visa Requirements</h2>
    <p class="section-subtitle">Click on a country to see swift details</p>
</div>
""", unsafe_allow_html=True)

# Mock Data for Map
data = {
    "Country": ["India", "United States", "Canada", "United Kingdom", "France", "Germany", "Japan", "Australia", "Thailand", "New Zealand"],
    "ISO": ["IND", "USA", "CAN", "GBR", "FRA", "DEU", "JPN", "AUS", "THA", "NZL"],
    "Visa Info": [
        "E-Visa available for many nationalities. 30-day tourist visa.",
        "ESTA required for visa-waiver countries. B1/B2 visa for others.",
        "eTA required for visa-exempt. Visitor visa for others.",
        "ETA implementation in progress. Standard Visitor Visa for most.",
        "Schengen Area rules apply. 90/180 days rule.",
        "Schengen Area rules apply. 90/180 days rule.",
        "Visa-free for many western countries up to 90 days.",
        "eVisitor (651) or ETA (601) required for many tourists.",
        "Visa Exemption for 60 days for many countries.",
        "NZeTA required for transit and tourism."
    ]
}
df = pd.DataFrame(data)

# Create Map
fig = px.choropleth(
    df,
    locations="ISO",
    locationmode="ISO-3",
    color="Country",
    hover_name="Country",
    hover_data={"Visa Info": True, "ISO": False, "Country": False},
    projection="natural earth",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Style the Map to look like SVG/Glass
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=0, b=0),
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showlakes=False,
        showocean=False,
        coastlinecolor="rgba(255,255,255,0.5)",
        countrycolor="rgba(255,255,255,0.1)",
        projection_type="natural earth"
    ),
    showlegend=False,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ----------------- INFO BOX / FOOTER -----------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 3rem; margin-bottom: 20px;">🤖</div>
        <h3 style="color: #00d2ff; margin: 0;">AI Visa Analyzer</h3>
        <p style="opacity: 0.8; margin-top: 10px;">
            Our advanced RAG pipeline reads official government documents to give you precise answers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 3rem; margin-bottom: 20px;">👤</div>
        <h3 style="color: #00d2ff; margin: 0;">Smart Profile</h3>
        <p style="opacity: 0.8; margin-top: 10px;">
            Save your passport details once and get personalized visa assessments instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 50px;"></div>
<div class="glass-card" style="padding: 20px; text-align: center; border-radius: 20px 20px 0 0; margin-bottom: 0;">
    <p style="color: white; font-weight: 600; margin-bottom: 5px;">SwiftVisa © 2025</p>
    <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)