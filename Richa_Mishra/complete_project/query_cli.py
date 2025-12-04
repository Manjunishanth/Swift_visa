import streamlit as st
import json
import os
from dotenv import load_dotenv
from rag.pipeline import run_rag

# --- Configuration ---
# Load environment variables (API keys, etc.)
load_dotenv()
QUERY_RESULTS_FILE = "query_results.json"
TOP_K_DISPLAY = 5

st.set_page_config(
    page_title="RAG Eligibility Assistant Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---

def load_results():
    """Loads the results from the batch processing JSON file."""
    if not os.path.exists(QUERY_RESULTS_FILE):
        st.error(f"Batch results file '{QUERY_RESULTS_FILE}' not found. Please run the batch processing script first.")
        return None
    try:
        with open(QUERY_RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from '{QUERY_RESULTS_FILE}'. File might be corrupted.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading results: {e}")
        return None

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
    
    # Simple placeholder for User Profile for future expansion
    # user_profile_json = st.text_area(
    #     "Optional: User Profile (JSON)", 
    #     value="{}",
    #     height=50
    # )

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

            # 3. Display Sources (Grounding Sources)
            sources = result.get("sources", [])
            st.subheader(f"Grounding Sources ({len(sources)})")
            if sources:
                for i, source in enumerate(sources):
                    st.markdown(
                        f"**{i+1}.** [{source.get('title', 'No Title')}]({source.get('uri', '#')})", 
                        help=source.get('uri', 'URI not available')
                    )
            else:
                st.info("No grounding sources were cited for this response.")

        except Exception as e:
            st.error(f"An error occurred during live query processing: {e}")
            st.exception(e)


def batch_results_tab():
    """Displays the results of the batch query processing."""
    st.header("📊 Batch Processing Results Viewer")
    st.markdown("Viewing results from the `query_results.json` file.")

    results_data = load_results()

    if results_data:
        st.subheader(f"Total Queries Processed: {len(results_data)}")
        
        # Convert results into a flat list of dictionaries for easier viewing
        flat_data = []
        for item in results_data:
            response = item.get("response", {})
            parsed = response.get("parsed", {}) # NEW: Access the structured 'parsed' object
            
            # Extract structured fields
            decision = parsed.get("decision", "N/A")
            reason = str(parsed.get("reason", "No detailed reason."))

            sources = response.get("sources", [])
            
            # Simple aggregation of source URIs for the table
            source_uris = "\n".join([s.get('uri', '') for s in sources])

            flat_data.append({
                "Query": item.get("query"),
                "Eligibility Status": decision,
                "Reason Summary": reason,
                "Num Sources": len(sources),
                "Source URIs": source_uris,
                "Confidence": parsed.get('confidence', 'N/A'),
                "Status": "✅ Success" if "parsed" in response else "❌ Error",
            })

        # Display as an interactive DataFrame
        st.dataframe(
            flat_data, 
            use_container_width=True, 
            column_config={
                "Query": st.column_config.TextColumn(width="medium"),
                "Eligibility Status": st.column_config.TextColumn("Status", width="small"),
                "Reason Summary": st.column_config.TextColumn("Reason", width="large"),
                "Source URIs": st.column_config.TextColumn("Sources", width="small"),
                "Num Sources": st.column_config.NumberColumn("Sources Count"),
                "Confidence": st.column_config.TextColumn("LLM Confidence"),
            },
            height=600
        )
        
        # Option to download the raw JSON
        st.download_button(
            label="Download Raw Results JSON",
            data=json.dumps(results_data, indent=2, ensure_ascii=False),
            file_name="query_results_download.json",
            mime="application/json"
        )


def main():
    """Main function to run the Streamlit app."""
    # Ensure the main title is always rendered, even if the subsequent logic fails.
    st.title("🌍 Eligibility RAG System Dashboard")
    st.markdown("A frontend tool to test the RAG pipeline and view batch processing outcomes for visa eligibility queries.")
    
    try:
        # Wrap the core logic to catch and display runtime errors gracefully
        tab1, tab2 = st.tabs(["Live Test Console", "Batch Results Viewer"])

        with tab1:
            live_query_tab()

        with tab2:
            batch_results_tab()

    except Exception as e:
        # This catch should only handle errors that occur *after* the initial
        # title rendering but *before* the tabs are fully functional.
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
        # In a real Streamlit environment, this console output is what you would need to check.