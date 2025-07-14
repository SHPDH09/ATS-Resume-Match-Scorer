import streamlit as st
import pandas as pd
import joblib
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import socket
import platform
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Google Sheets Logging ---
def append_to_google_sheet(data):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1k3Iys5lD26VjqNUGyhB2FlK4NSkDN4vHaHMyKvNPR5g").sheet1
    sheet.append_row(data)

def log_visitor_info():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except:
        ip = "Unknown"
    sys_name = platform.node()
    sys_info = platform.platform()
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = "Mobile" if any(x in sys_info.lower() for x in ["android", "iphone"]) else "PC"
    row = [time, ip, sys_name, sys_info, device]
    try:
        append_to_google_sheet(row)
        st.success("✅ Visitor logged successfully.")
    except Exception as e:
        print("Logging error:", e)

log_visitor_info()

# --- Visitor Counter ---
def update_visit_count():
    count_file = "visits.txt"
    if not os.path.exists(count_file):
        with open(count_file, "w") as f:
            f.write("0")
    with open(count_file, "r+") as f:
        count = int(f.read().strip())
        count += 1
        f.seek(0)
        f.write(str(count))
        f.truncate()
    return count

visit_count = update_visit_count()

# --- Streamlit Config ---
st.set_page_config(page_title="ATS Resume Match", layout="wide")

st.markdown(f"""
    <style>
        header {{visibility: hidden;}}
        .custom-header {{
            display: flex; justify-content: space-between;
            background-color: #f0f2f6; padding: 1rem 2rem;
            border-radius: 10px; margin-bottom: 20px;
        }}
        .header-title {{ font-size: 24px; font-weight: bold; }}
    </style>
    <div class="custom-header">
        <div class="header-title">📄 ATS Resume Match Scorer</div>
        <div><span style='font-size:16px;'>👀 Total Visits: <b>{visit_count}</b></span></div>
    </div>
""", unsafe_allow_html=True)

# --- Toggle Buttons ---
if "show_about" not in st.session_state: st.session_state.show_about = False
if "show_contact" not in st.session_state: st.session_state.show_contact = False
col1, col2, col3 = st.columns([5, 1, 1])
with col2:
    if st.button("📘 About"): st.session_state.show_about, st.session_state.show_contact = not st.session_state.show_about, False
with col3:
    if st.button("📬 Contact"): st.session_state.show_contact, st.session_state.show_about = not st.session_state.show_contact, False

# --- Theme Switch ---
theme = st.selectbox("🌗 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            .main { background-color: #1e1e1e; color: white; }
            div.stButton > button { color: black; }
        </style>
    """, unsafe_allow_html=True)

# --- About / Contact ---
if st.session_state.show_about:
    with st.expander("📘 About This App", expanded=True):
        st.markdown("""
        This ATS Resume Matcher helps compare your resume with job descriptions using **TF-IDF** and **Cosine Similarity**.
        Upload your resume, and we’ll show the most relevant jobs or internships.
        """)

if st.session_state.show_contact:
    with st.expander("📬 Contact Me", expanded=True):
        st.markdown("""
        **👨‍💻 Developer:** Raunak Kumar  
        📧 Email: raunakkumarjob@gmail.com  
        🔗 [LinkedIn](https://linkedin.com/in/your-link)
        """)

# --- Load Resources ---
@st.cache_resource
def load_vectorizer(): return joblib.load("vectorizer.pkl")

@st.cache_data
def load_job_dataset():
    df = pd.read_csv("careers.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

vectorizer = load_vectorizer()
career_data = load_job_dataset()

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "".join([p.extract_text() or "" for p in reader.pages])

def get_ats_score(resume_text, job_description):
    vect_resume = vectorizer.transform([resume_text])
    vect_job = vectorizer.transform([job_description])
    similarity = cosine_similarity(vect_resume, vect_job)[0][0]
    return round(similarity * 100, 2)

# --- Main Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📤 Upload Your Resume")
    resume_file = st.file_uploader("PDF Only", type=["pdf"])
    resume_text = extract_text_from_pdf(resume_file) if resume_file else ""
    if resume_text:
        with st.expander("📄 View Resume Text"):
            st.text(resume_text[:3000])

with col2:
    if resume_file:
        st.subheader("⚙️ Select Your Preference")
        job_type = st.selectbox("Choose Type", ["Job", "Internship"])
        filtered_data = pd.DataFrame()

        if job_type == "Job":
            exp_type = st.radio("Experience Level", ["Fresher", "Experienced"])
            if "experience" in career_data.columns:
                if exp_type == "Fresher":
                    filtered_data = career_data[
                        (career_data["type"].str.lower() == "job") &
                        (career_data["experience"].str.lower() == "fresher")
                    ]
                else:
                    filtered_data = career_data[
                        (career_data["type"].str.lower() == "job") &
                        (career_data["experience"].str.lower() != "fresher")
                    ]
        else:
            filtered_data = career_data[career_data["type"].str.lower() == "internship"]

        if st.button("🔍 Match Resume to Roles"):
            vect_resume = vectorizer.transform([resume_text])
            scores = []
            for _, row in filtered_data.iterrows():
                vect_job = vectorizer.transform([row["description"]])
                sim = cosine_similarity(vect_resume, vect_job)[0][0]
                scores.append({
                    "title": row["title"],
                    "company": row["company"],
                    "score": round(sim * 100, 2),
                    "apply_link": row["apply_link"]
                })
            top_matches = sorted(scores, key=lambda x: x["score"], reverse=True)[:10]
            if top_matches:
                st.subheader("📌 Top Matches")
                for job in top_matches:
                    st.markdown(f"""
                    ### 🧑‍💼 {job['title']}
                    **🏢 {job['company']}**  
                    🔗 [Apply Now]({job['apply_link']})  
                    📊 **ATS Score: {job['score']}%**  
                    ---""")
            else:
                st.warning("No roles matched your resume.")

        st.subheader("📊 General ATS Score (Custom JD)")
        with st.expander("📝 Paste Job Description"):
            custom_jd = st.text_area("Enter job description")
            if st.button("🎯 Check ATS Score"):
                if custom_jd.strip():
                    score = get_ats_score(resume_text, custom_jd)
                    st.success(f"✅ Match Score: **{score}%**")
                else:
                    st.warning("Please enter a job description.")
    else:
        st.warning("📌 Please upload your resume to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Made with ❤️ by <b>Raunak Kumar</b> | © 2025<br>"
    "This tool is for educational use only."
    "</div>", unsafe_allow_html=True
)
