import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="ATS Resume Match", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 ATS Resume Match Scorer</h1>", unsafe_allow_html=True)

# --- Session State for popups ---
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "show_contact" not in st.session_state:
    st.session_state.show_contact = False

# --- HEADER with About and Contact Buttons ---
colA1, colA2, colA3 = st.columns([5, 1, 1])
with colA2:
    if st.button("📘 About"):
        st.session_state.show_about = not st.session_state.show_about
        st.session_state.show_contact = False

with colA3:
    if st.button("📬 Contact"):
        st.session_state.show_contact = not st.session_state.show_contact
        st.session_state.show_about = False

# --- ABOUT Popup ---
if st.session_state.show_about:
    with st.expander("📘 About This App", expanded=True):
        st.markdown("""
        This ATS Resume Matcher helps you compare your resume with job or internship descriptions.  
        Using **Machine Learning (TF-IDF + Cosine Similarity)**, it calculates a match score and recommends the best-fit roles.

        🔍 Features:
        - Upload Resume (PDF)
        - Match with Job/Internship roles
        - Get ATS Match Score
        - See Top Recommended Positions
        """)

# --- CONTACT Popup ---
if st.session_state.show_contact:
    with st.expander("📬 Contact Me", expanded=True):
        st.write("**👨‍💻 Developer:** Raunak Kumar")
        st.write("📧 Email: raunakkumarjob@gmail.com")
        st.write("🔗 LinkedIn: [Raunak's Profile](https://linkedin.com/in/your-link)")

# --- Load vectorizer ---
@st.cache_resource
def load_vectorizer():
    return joblib.load("vectorizer.pkl")

# --- Load and clean job dataset ---
@st.cache_data
def load_job_dataset():
    df = pd.read_csv("careers.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

# --- PDF Resume Text Extractor ---
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- ATS Score Calculation ---
def get_ats_score(resume_text, job_description):
    vect_resume = vectorizer.transform([resume_text])
    vect_job = vectorizer.transform([job_description])
    similarity = cosine_similarity(vect_resume, vect_job)[0][0]
    return round(similarity * 100, 2)

# --- Load Data ---
vectorizer = load_vectorizer()
career_data = load_job_dataset()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Your Resume")
    resume_file = st.file_uploader("PDF Only", type=["pdf"])
    resume_text = ""

    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        with st.expander("📄 View Extracted Resume Text"):
            st.text(resume_text[:3000])  # Preview

with col2:
    if resume_file:
        st.subheader("⚙️ Select Your Preference")
        job_type = st.selectbox("Choose Type", ["Job", "Internship"])
        filtered_data = pd.DataFrame()

        if job_type.lower() == "job":
            experience_type = st.radio("Are you a:", ["Fresher", "Experienced"])
            if "experience" in career_data.columns:
                if experience_type == "Fresher":
                    filtered_data = career_data[
                        (career_data["type"].str.lower() == "job") &
                        (career_data["experience"].str.lower() == "fresher")
                    ]
                else:
                    filtered_data = career_data[
                        (career_data["type"].str.lower() == "job") &
                        (career_data["experience"].str.lower() != "fresher")
                    ]
        elif job_type.lower() == "internship":
            filtered_data = career_data[
                career_data["type"].str.lower() == "internship"
            ]

        if st.button("🔍 Match Resume to Roles"):
            if resume_text.strip():
                vect_resume = vectorizer.transform([resume_text])
                scores = []

                for _, row in filtered_data.iterrows():
                    job_desc = row["description"]
                    vect_job = vectorizer.transform([job_desc])
                    similarity = cosine_similarity(vect_resume, vect_job)[0][0]
                    score = round(similarity * 100, 2)

                    scores.append({
                        "title": row['title'],
                        "company": row['company'],
                        "score": score,
                        "apply_link": row['apply_link']
                    })

                sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

                if sorted_scores:
                    st.subheader("📌 Top Matching Roles")
                    top_n = st.slider("How many results to show?", 5, 50, 10)

                    for i in range(0, min(len(sorted_scores), top_n), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i + j < len(sorted_scores):
                                job = sorted_scores[i + j]
                                with cols[j]:
                                    st.markdown(f"### 🧑‍💼 {job['title']}")
                                    st.markdown(f"🏢 **{job['company']}**")
                                    st.markdown(f"🔗 [Apply Now]({job['apply_link']})", unsafe_allow_html=True)
                                    st.success(f"📊 ATS Score: **{job['score']}%**")
                                    st.markdown("---")

                    best_match = sorted_scores[0]
                    st.info(f"✅ **Top Recommendation:** {best_match['title']} at {best_match['company']} ({best_match['score']}%)")
                else:
                    st.warning("⚠️ No jobs matched your resume.")
            else:
                st.warning("⚠️ Resume content is empty.")
        elif filtered_data.empty:
            st.warning("⚠️ No roles matched your criteria.")

        # 🔍 General JD ATS Score
        st.subheader("📊 General ATS Score (Custom JD)")
        with st.expander("📝 Paste Custom Job Description"):
            custom_jd = st.text_area("Enter job description here...")
            if st.button("🎯 Check General ATS Score"):
                if custom_jd.strip() and resume_text.strip():
                    score = get_ats_score(resume_text, custom_jd)
                    st.success(f"✅ Your Resume ATS Match Score: **{score}%**")
                else:
                    st.warning("📄 Resume or Job Description is missing.")
    else:
        st.warning("📌 Please upload your resume to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Made with ❤️ by <b>Raunak Kumar</b> | © 2025 All Rights Reserved<br>"
    "This tool is for educational use only. Do not use for real-time hiring decisions."
    "</div>", unsafe_allow_html=True
)
