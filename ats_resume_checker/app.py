import os
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(
    page_title="ATS Resume Checker",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Page styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 ATS Resume Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your resume, paste the job description, and get an AI-powered ATS evaluation.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# API key
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "Google API Key",
        value=api_key or "",
        type="password",
        help="You can also store GOOGLE_API_KEY in a .env file."
    )

if not api_key_input:
    st.warning("Enter your Google API key in the sidebar to use the checker.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key_input

# -----------------------------
# User inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader(
        "Upload your resume as PDF",
        type=["pdf"]
    )

with col2:
    st.subheader("2. Job Description")
    description = st.text_area(
        "Paste the complete job description here",
        height=300,
        placeholder=(
            "Example:\n"
            "Job Title: Data Scientist\n"
            "Skills Required: Python, SQL, Machine Learning, Pandas..."
        )
    )

check_button = st.button(
    "🔍 Check ATS Score",
    type="primary",
    use_container_width=True
)

# -----------------------------
# Main logic
# -----------------------------
if check_button:
    if resume_file is None:
        st.error("Please upload a PDF resume.")
        st.stop()

    if not description.strip():
        st.error("Please enter the job description.")
        st.stop()

    temp_resume = "uploaded_resume.pdf"

    try:
        with open(temp_resume, "wb") as f:
            f.write(resume_file.getbuffer())

        # Same PDF -> text approach used in the notebook.
        loader = PyPDFLoader(temp_resume)
        content = loader.load()

        pages = []
        for page in content:
            pages.append(page.page_content)

        text = "\n".join(pages)

        if not text.strip():
            st.error("Could not extract text from the PDF. Try a text-based PDF.")
            st.stop()

        # Same prompts/chain structure as the uploaded notebook.
        ats_score_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Act as an ATS expert with 40 years of experience in ATS "
                    "score checking. Check the ATS score based on the given "
                    "job description. Return an ATS score as a percentage. "
                    "If any mistakes or gaps are found, clearly return them "
                    "with useful suggestions for improvement."
                ),
                (
                    "human",
                    "resume:\n{resume}\n\n"
                    "description:\n{description}"
                ),
            ]
        )

        eligible_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Act as a senior human resource manager. Check the resume "
                    "against the given requirements. If the ATS score is more "
                    "than 70% and the resume matches the requirements, return "
                    "'ELIGIBLE / SHORTLISTED'. Otherwise return "
                    "'NOT SHORTLISTED'. Also explain the important matching "
                    "and missing requirements."
                ),
                (
                    "human",
                    "resume:\n{resume}\n\n"
                    "description:\n{description}"
                ),
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.2,
            google_api_key=api_key_input
        )

        parser = StrOutputParser()

        ats_chain = ats_score_prompt_template | llm | parser
        eligible_chain = eligible_prompt_template | llm | parser

        full_chain = RunnableParallel(
            ats_score=ats_chain,
            eligibility=eligible_chain
        )

        with st.spinner("Analyzing your resume..."):
            response = full_chain.invoke(
                {
                    "resume": text,
                    "description": description
                }
            )

        st.success("ATS analysis completed!")

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.subheader("📊 ATS Score Analysis")
            st.markdown(
                f'<div class="result-card">{response["ats_score"]}</div>',
                unsafe_allow_html=True
            )

        with result_col2:
            st.subheader("👤 Eligibility Analysis")
            st.markdown(
                f'<div class="result-card">{response["eligibility"]}</div>',
                unsafe_allow_html=True
            )

        st.divider()

        st.subheader("📋 Resume Text Extracted")
        with st.expander("Show extracted resume text"):
            st.text(text)

    except Exception as e:
        st.error("Something went wrong while checking the resume.")
        st.exception(e)

    finally:
        if os.path.exists(temp_resume):
            os.remove(temp_resume)
