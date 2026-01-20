import streamlit as st
import json
import os
from dotenv import load_dotenv
from rag.pipeline import run_rag

# --- Configuration ---
load_dotenv()
QUERY_RESULTS_FILE = "query_results.json"
TOP_K_DISPLAY = 5

st.set_page_config(
    page_title="SwiftVisa AI - Your AI Eligibility Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
    }
    
    /* Content container with glassmorphism effect */
    .block-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e3c72;
        font-weight: 600;
        border-left: 4px solid #2a5298;
        padding-left: 1rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Paragraph text */
    p {
        color: #2c2c2c;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #2c2c2c;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Flag banner */
    .flag-banner {
        color: #1e3c72;
        font-weight: 700;
        text-align: center;
        font-size: 3rem;
        margin: 1rem 0 2rem 0;
        letter-spacing: 0.5rem;
        animation: wave 3s ease-in-out infinite;
    }
    
    @keyframes wave {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #1e3c7215 0%, #2a529815 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 20px rgba(30, 60, 114, 0.3);
    }
    
    .feature-card h3 {
        color: #1a1a1a !important;
    }
    
    .feature-card p {
        color: #2c2c2c !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(30, 60, 114, 0.6);
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 15px;
        border: 2px solid #1e3c7230;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.2);
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left-width: 5px;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #0f9b0f 0%, #2ecc71 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(15, 155, 15, 0.3);
    }
    
    /* Status badges */
    .status-eligible {
        color: #0f9b0f;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    .status-not-eligible {
        color: #c0392b;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    /* Document chips */
    .doc-chip {
        display: inline-block;
        background: #1e3c7220;
        color: #1e3c72;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Spinner container */
    .stSpinner > div {
        border-top-color: #2a5298 !important;
    }
    
    /* Info box with icon */
    .info-box {
        background: linear-gradient(135deg, #1e3c7210 0%, #2a529810 100%);
        border-left: 4px solid #2a5298;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    
    .info-box strong {
        color: #1a1a1a !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---

def format_llm_response_streamlit(parsed: dict) -> str:
    """Formats the structured LLM response into a readable Markdown string."""
    def normalize(value):
        if value is None:
            return "Not Provided"
        if isinstance(value, list):
            filtered_list = [v for v in value if str(v).lower() not in ("null", "not provided", "")]
            return "\n- " + "\n- ".join(str(v) for v in filtered_list) if filtered_list else "None"
        if isinstance(value, dict):
            return "\n" + "\n".join(f"- {k}: {v}" for k, v in value.items())
        return str(value)

    decision = parsed.get("decision")
    reason = parsed.get("reason")
    
    if not any([decision, reason]):
        return parsed.get("raw") or "**LLM returned empty structured data.**\n" + json.dumps(parsed, indent=2)

    # Create status badge based on decision
    decision_text = str(decision).lower()
    status_class = "status-eligible" if "eligible" in decision_text or "yes" in decision_text else "status-not-eligible"
    
    lines = [
        f"<p class='{status_class}' style='color: #1a1a1a;'>📋 Eligibility Status: {normalize(decision)}</p>",
        f"<p style='color: #1a1a1a;'><strong>🎯 Confidence Score:</strong> {normalize(parsed.get('confidence'))}</p>",
        f"<p style='color: #1a1a1a;'><strong>💡 Reason for Decision:</strong> {normalize(reason)}</p>",
        f"<p style='color: #1a1a1a;'><strong>🚀 Actions to Improve:</strong> {normalize(parsed.get('future_steps'))}</p>", 
    ]
    return "\n\n".join(lines)


# --- Application Components ---

def render_hero_section():
    """Renders the hero section with flags and title."""
    st.markdown('<div class="flag-banner">USA UK CANADA IRELAND SCHENGEN</div>', unsafe_allow_html=True)
    st.title("SwiftVisa AI")
    st.markdown('<p class="subtitle">✨ Your Intelligent Visa Eligibility Assistant - Powered by AI ✨</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Instant Analysis</h3>
            <p>Get visa eligibility results in seconds with our advanced AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Accurate Results</h3>
            <p>AI-powered analysis based on official immigration guidelines</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🌐 Multiple Countries</h3>
            <p>Support for visa applications across various countries</p>
        </div>
        """, unsafe_allow_html=True)


def live_query_tab():
    """Allows user to input a single query and run the RAG pipeline live."""
    
    st.markdown("---")
    st.markdown("<h2 style='color:#1a1a1a;'>🧪 Try Your Eligibility Check</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💬 How it works:</strong> Enter your visa query below and our AI will analyze your eligibility 
        based on official immigration requirements and provide personalized guidance.
    </div>
    """, unsafe_allow_html=True)

    query = st.text_area(
        "✍️ Enter your visa eligibility query:", 
        placeholder="Example: Am I eligible for a UK Spouse Visa if my partner earns £29,000 per year?",
        height=120,
        help="Be as specific as possible for the most accurate results"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🔍 Analyze My Eligibility", type="primary", use_container_width=True)
    
    if run_button:
        if not query:
            st.warning("⚠️ Please enter a query to run the RAG pipeline.")
            return

        try:
            with st.spinner("🔄 Processing your query and analyzing eligibility..."):
                result = run_rag(query, top_k=TOP_K_DISPLAY)
            
            st.success("✅ Analysis completed successfully!")
            
            parsed = result.get("parsed", {})
            retrieved = result.get("retrieved", [])
            final_confidence = result.get('final_confidence', 0.0)

            # Display results in an attractive layout
            st.markdown("---")
            
            # 1. Structured Answer with icon
            
            st.markdown("<h3 style='color:#1a1a1a;'>🗣️ AI Assistant - Your Eligibility Analysis</h3>", unsafe_allow_html=True)
            st.markdown(format_llm_response_streamlit(parsed), unsafe_allow_html=True)
            
            st.markdown("---")

            # 2. Metrics Display
            st.markdown("<h3 style='color:#1a1a1a;'>📊 Analysis Metrics</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_percent = f"{final_confidence * 100:.1f}%"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{confidence_percent}</div>
                    <div>Based on {len(retrieved)} official sources</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Documents Analyzed</div>
                    <div class="metric-value">{TOP_K_DISPLAY}</div>
                    <div>Official immigration guidelines</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. Document References
            if retrieved:
                st.markdown("---")
                st.markdown("<h3 style='color:#1a1a1a;'>📑 Referenced Documents</h3>", unsafe_allow_html=True)
                st.markdown("*Sources used for this analysis:*")
                
                doc_chips = "".join([f'<span class="doc-chip">📄 Doc #{c.get("uid", "N/A")}</span>' 
                                    for c in retrieved[:TOP_K_DISPLAY]])
                st.markdown(doc_chips, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {e}")
            st.exception(e)


def render_sidebar():
    """Renders an attractive sidebar with information."""
    with st.sidebar:
        st.markdown("## 🎯 Quick Guide")
        st.markdown("""
        ### How to Use:
        1. 📝 **Enter your query** about visa eligibility
        2. 🔍 **Click analyze** to get AI-powered results
        3. 📊 **Review** your eligibility status and recommendations
        
        ### Supported Visa Types:
        - 💼 Work Visas
        - 👨‍👩‍👧 Family/Spouse Visas
        - 🎓 Student Visas
        - 🏖️ Tourist Visas
        - 💰 Investment Visas
        
        ### Countries Covered:
        🇬🇧 United Kingdom
        🇺🇸 United States
        🇨🇦 Canada
        🇦🇺 Australia
        🇪🇺 European Union
        *and many more...*
        
        ---
        
        ### 💡 Tips for Best Results:
        - Be specific about your situation
        - Include relevant financial details
        - Mention your relationship status
        - State your nationality
        
        ---
        
        ### ⚠️ Disclaimer:
        This tool provides guidance based on AI analysis. Always consult official immigration authorities for final decisions.
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Need Help?")
        st.markdown("Contact our support team for personalized assistance.")


def main():
    """Main function to run the Streamlit app."""
    
    try:
        render_sidebar()
        render_hero_section()
        live_query_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>🌟 Made with ❤️ by SwiftVisa AI Team | Powered by Advanced AI Technology</p>
            <p style="font-size: 0.9rem;">© 2024 SwiftVisa AI. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ A critical runtime error occurred while executing dashboard components.")
        st.exception(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"--- CRITICAL STREAMLIT STARTUP FAILURE ---")
        print(f"The application failed to import or initialize: {e}")
        print(f"Check your dependencies (pip install -r requirements.txt) and your API key configuration in .env.")
        print("------------------------------------------")