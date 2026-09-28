# ATS Resume Checker

A simple Streamlit website based on the uploaded `Ats_score(2).ipynb`.

## What it does

1. User uploads a PDF resume.
2. User pastes a job description.
3. The app extracts text from the PDF.
4. The same LangChain idea from the notebook is used:
   - ATS score analysis
   - Eligibility / shortlisted analysis
5. Results are displayed in the browser.

## Run locally

Open a terminal in this folder:

```bash
pip install -r requirements.txt
```

Set your API key either in a `.env` file:

```text
GOOGLE_API_KEY=your_api_key
```

or enter it in the app sidebar.

Then run:

```bash
streamlit run app.py
```

The browser will open the ATS Resume Checker.

## Important

The notebook uses `ChatGoogleGenerativeAI` and the model name `gemini-3.5-flash`, so the app keeps that same model configuration. If that model is unavailable for your Google API account, change the `model=` value in `app.py` to a model available to your account.
