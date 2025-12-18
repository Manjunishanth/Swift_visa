### SwiftVisa – AI Visa Eligibility Screening Agent

SwiftVisa is an AI-powered visa eligibility screening system built using Retrieval-Augmented Generation (RAG).
It helps users quickly assess whether they are eligible for a visa based on their profile, documents, and financial details, using real visa policy documents.

The system provides:

-->Eligibility decision (YES / NO)

-->Confidence score (0–1)

-->Clear explanation (reason)

-->History of previous checks

### Project Goal
The goal of SwiftVisa is to automate the initial visa screening process by:

-->Reducing dependency on manual consultations

-->Using policy-backed AI decisions

-->Providing fast, explainable, and user-friendly outputs

-->Demonstrating a real-world RAG + LLM application

### Supported Countries

---United States

---United Kingdom

---Canada

---Schengen

---Ireland

### Supported Visa Types

---Student

---Tourist

---Business

---Work

---Dependent

---Family 

### Streamlit Application

SwiftVisa is built as a menu-driven Streamlit application with multiple pages to ensure a clean, user-friendly experience.
Users can navigate between pages using the sidebar menu.

I started by configuring the application layout using st.set_page_config() to define the page title, icon, and centered layout. 
This helped maintain a clean and consistent appearance throughout the application.

To organize the application, I implemented a sidebar navigation menu using st.sidebar.radio(). 
The sidebar allows users to switch between different sections of the app, including Home, Visa Eligibility Check, History, and About, without reloading the application.

 ## Home Page
  
  The Home page serves as the introduction to SwiftVisa.
  
  ## Purpose:
  
  ---Welcomes users to the application
  
  ---Explains what SwiftVisa does in simple terms
  
  ---Guides users on how to navigate the app
  
  This page is ideal for first-time users to understand the system before using it.

 ## Check Eligibility Page
  
  This is the core functional page of the application.
  
  For collecting user inputs, I used Streamlit forms (st.form) along with input components such as text inputs, number inputs, select boxes, and text areas. 
  Grouping inputs inside a form ensured that the data is processed only after the user clicks the submit button, improving performance and user experience.

  During the eligibility evaluation process, I integrated Streamlit spinners (st.spinner) to indicate that the AI model and retrieval logic are processing the request.
  Once the response is generated, the results are displayed using Streamlit’s visual elements such as st.success() and st.error() for visa approval or rejection, 
  st.metric() for eligibility status and confidence score, and st.progress() to visually represent the confidence level.
 
  ## Purpose:
  
  --Collect user visa-related details
  
  --Perform AI-based eligibility analysis
  
  --Display the final decision

  ## User Inputs Collected:
 
  --Full Name
  
  --Age
  
  --Nationality
  
  --Destination Country
  
  --Visa Type
  
  --Education / Job details
  
  --Financial proof
  
  --Available documents
  
  --Case explanation
  
  ## Outputs Displayed:
  
  --Eligibility decision (YES / NO)
  
  --Confidence score (0–1) with progress bar
  
  --Clear explanation (reason)
 
 ## History Page
 
 The History page shows previously checked visa cases. 
 To maintain previously checked visa results, I used Streamlit session state (st.session_state) to store user history during the session.
 This allowed me to implement a History page without using an external database.
 
  ## Purpose:
  
  --Track earlier eligibility checks
  
  --Allow users to review past decisions
  
  --Improve transparency and usability
  
  ## Session-Based Storage:
  
  --History is stored in Streamlit session state
  
  --Data is cleared once the session ends
  
  --No permanent storage of personal data
 
 ## About Page
 
 The About page explains the project in detail.
 
 ## Purpose:
 
  --Describe what SwiftVisa is
  
  --Explain how it works internally
  
  --Highlight key features and benefits

### Navigation Flow

Sidebar Menu
 ├── Home
 ├── Check Eligibility
 ├── History
 └── About

Users can switch between pages without losing session data

### Installation

1.Install Dependencies

pip install -r requirements.txt

2.Set Environment Variable

export GROQ_API_KEY="your_api_key"

3.Run the App

streamlit run app.py

### Disclaimer

SwiftVisa provides preliminary eligibility guidance only.
Final visa decisions are made exclusively by official embassies or immigration authorities.
  
