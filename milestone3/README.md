# Milestone 3 — Web app (Streamlit)

This folder contains the web application for interactive visa analysis and user profiles.

## Contents
- `app.py` — application entry (currently empty in this repo; Streamlit will discover pages in the `pages/` folder).
- `pages/` — Streamlit multi-page components:
  - `2_UserProfile.py` — user profile form and session-state storage
  - `3_Visa_Analyzer.py` — RAG-powered visa question analyzer (calls `src/rag_pipeline.answer_question`)

## Requirements
- Install project dependencies (from the repo root):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements
```

Ensure `streamlit` and any model/embedding dependencies referenced in `src/` are installed.

## Run the app
From the repository root run:

```powershell
streamlit run milestone3/app.py
```

Streamlit will load `app.py` and the pages under `milestone3/pages/`. Use the sidebar to navigate between **Home**, **User Profile**, and **Visa Analyzer**.

## Notes & troubleshooting
- `pages/3_Visa_Analyzer.py` imports `src.rag_pipeline.answer_question` — ensure `src/` and its dependencies are available and configured (models, `milestone1/models/faiss.index`).
- If `app.py` is empty, Streamlit still shows the pages; to create a landing/home page, add content to `app.py` or update the sidebar links in the `pages` files.
- Logs and errors will appear in the terminal where you run Streamlit.

## Next steps
- Add a simple `app.py` homepage or combine navigation into a single file.
- Confirm `src/` dependencies and optionally provide a sample `.env` for keys/config.

If you want, I can: add a small `app.py` home page, add a `run_app.ps1` convenience script, or create a minimal `.env.example`.
