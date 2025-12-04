import streamlit as st
import json
import os
from dotenv import load_dotenv
from rag.pipeline import run_rag

# --- Configuration ---
# Load environment variables (API keys, etc.)
load_dotenv()
QUERY_RESULTS_FILE = "query_results.json" # Kept, but no longer used in the main app logic
TOP_K_DISPLAY = 5

st.set_page_config(
    page_title="SwiftVisa AI - Your AI Eligibility Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---

def format_llm_response_streamlit(parsed: dict) -> str:
    """
    Formats the structured LLM response (from the 'parsed' key) into a readable Markdown string,
    mirroring the logic from the command-line interface formatter.
    """
    def normalize(value):
        if value is None:
            return "Not Provided"
        if isinstance(value, list):
            # Check for empty list after stripping potential 'null' strings
            filtered_list = [v for v in value if str(v).lower() not in ("null", "not provided", "")]
            return "\n- " + "\n- ".join(str(v) for v in filtered_list) if filtered_list else "None"
        if isinstance(value, dict):
            return "\n" + "\n".join(f"- {k}: {v}" for k, v in value.items())
        return str(value)

    decision = parsed.get("decision")
    reason = parsed.get("reason")
    
    # Fallback if critical structured fields are missing
    if not any([decision, reason]):
        return parsed.get("raw") or "**LLM returned empty structured data.**\n" + json.dumps(parsed, indent=2)

    lines = [
        f"**Eligibility Status:** {normalize(decision)}",
        f"**Confidence Score:** {normalize(parsed.get('confidence'))}",
        f"**Reason for Decision:** {normalize(reason)}",
        f"**Actions to Improve:** {normalize(parsed.get('future_steps'))}", 
    ]
    return "\n\n".join(lines)


# --- Application Components ---

def live_query_tab():
    """Allows user to input a single query and run the RAG pipeline live."""
    st.header("🧪 Live RAG Query Test")
    st.markdown("Test the RAG pipeline instantly with a custom query.")

    query = st.text_area(
        "Enter your visa eligibility query:", 
        placeholder="Am I eligible for a UK Spouse Visa if my partner earns £29,000?",
        height=100
    )
    
    if st.button("Run RAG Query", type="primary"):
        if not query:
            st.warning("Please enter a query to run the RAG pipeline.")
            return

        try:
            with st.spinner("Processing query and fetching grounded response..."):
                # Call the core RAG function
                result = run_rag(query, top_k=TOP_K_DISPLAY)
            
            st.success("Query processed successfully!")
            
            parsed = result.get("parsed", {})
            retrieved = result.get("retrieved", [])
            final_confidence = result.get('final_confidence', 0.0)

            # 1. Display Structured Answer
            st.subheader("🗣️ SwiftVisa Assistant - Structured Analysis")
            st.markdown(format_llm_response_streamlit(parsed))

            # 2. Display Retrieval Metadata
            st.subheader("📑 Retrieval Metadata")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Final Blended Confidence:** `{final_confidence:.3f}`")

            with col2:
                uids = [str(c.get("uid", "No UID")) for c in retrieved[:TOP_K_DISPLAY]]
                if uids:
                    st.markdown(f"**Chunks Used (Top {TOP_K_DISPLAY} UIDs):** `{', '.join(uids)}`")
                else:
                    st.markdown("**Chunks Used (Top UIDs):** `No relevant documents were retrieved.`")

        except Exception as e:
            st.error(f"An error occurred during live query processing: {e}")
            st.exception(e)


def main():
    """Main function to run the Streamlit app."""
    # Renamed title and updated description as requested
    st.title("SwiftVisa AI - Your AI Eligibility Assistant")
    st.markdown("A tool for instant eligibility analysis.")
    
    try:
        # We only call the live query function now, removing the need for st.tabs
        live_query_tab()

    except Exception as e:
        # This catch should only handle errors that occur *after* the initial
        # title rendering but *before* the UI is fully functional.
        st.error("A critical runtime error occurred while executing dashboard components.")
        st.exception(e)


if __name__ == "__main__":
    # CRITICAL DEBUGGING STEP: 
    # Wrap the entire execution in case the failure is during imports (e.g., missing 'rag.pipeline')
    try:
        main()
    except Exception as e:
        # This will only be reached if the failure is so early that Streamlit's internal error
        # handling can't display it. We print a warning to the console.
        print(f"--- CRITICAL STREAMLIT STARTUP FAILURE ---")
        print(f"The application failed to import or initialize: {e}")
        print(f"Check your dependencies (pip install -r requirements.txt) and your API key configuration in .env.")
        print("------------------------------------------")