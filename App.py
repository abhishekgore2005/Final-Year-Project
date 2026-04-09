import streamlit as st
import pypdf
import smtplib
import re
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Screener Pro", layout="wide")

# --- 2. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('resume_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candidates
                 (date TEXT, filename TEXT, email TEXT, score REAL, status TEXT, missing_skills TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(data_list):
    conn = sqlite3.connect('resume_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in data_list:
        c.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)", 
                  (timestamp, row['Filename'], row['Email'], row['Score'], row['Status'], row['Missing Skills']))
    conn.commit()
    conn.close()

init_db()

# --- 3. HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text.lower()
    except: return ""

def calculate_hybrid_score(resume_text, required_skills):
    if not resume_text: return 0.0, required_skills
    
    # --- PART A: NLP Scoring (50% weight) ---
    jd = " ".join(required_skills)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd])
        nlp_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    except: nlp_score = 0.0

    # --- PART B: Exact Match Scoring (50% weight) ---
    found_keywords = [s for s in required_skills if s.lower() in resume_text]
    keyword_score = (len(found_keywords) / len(required_skills)) * 100 if required_skills else 0
    
    # --- PART C: Combined Final Score ---
    final_score = round((nlp_score * 0.5) + (keyword_score * 0.5), 2)
    
    missing = [s for s in required_skills if s.lower() not in resume_text]
    return final_score, missing

def send_email(to_email, subject, body, s_email, s_pass):
    try:
        msg = MIMEMultipart()
        msg['From'] = f"HR Team <{s_email}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(s_email, s_pass)
        server.sendmail(s_email, to_email, msg.as_string())
        server.quit()
        return True
    except: return False

# --- 4. UI ---
st.title("🚀 AI Resume Screener (Hybrid Engine)")

with st.sidebar:
    st.header("⚙️ Settings")
    req_skills_input = st.text_area("Required Skills (Comma separated)", "python, sql, machine learning")
    cutoff = st.slider("Pass Cutoff Score (%)", 0, 100, 40)
    
    st.divider()
    st.header("📧 Email Automation")
    enable_email = st.checkbox("Enable Auto-Response")
    s_email = st.text_input("Sender Email", value="hirebot.project@gmail.com")
    s_pass = st.text_input("App Password", type="password", value="nfyq ghye qzlw bmcb")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Start Hybrid Analysis"):
    results = []
    skills_list = [s.strip().lower() for s in req_skills_input.split(",")]
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        email = email_match.group(0) if email_match else None
        
        score, missing = calculate_hybrid_score(text, skills_list)
        status = "SELECTED" if score >= cutoff else "REJECTED"
        
        # --- FILTER: EMAIL ONLY SELECTED ---
        if not enable_email:
            e_status = "Disabled"
        elif status == "REJECTED":
            e_status = "Skipped (Rejected)"
        elif not email:
            e_status = "Email Not Found ⚠️"
        elif status == "SELECTED" and s_email and s_pass:
            body = f"""
            <html>
                <body>
                    <h3>Interview Invitation</h3>
                    <p>Hello,</p>
                    <p>Congratulations! Your resume has been shortlisted by our Hybrid AI system.</p>
                    <p><b>Hybrid Match Score:</b> {score}%<br>
                    <b>Status:</b> {status}</p>
                    <p>Our team will reach out shortly for the interview schedule.</p>
                    <p>Best regards,<br>HR Recruitment Team</p>
                </body>
            </html>
            """
            e_status = "Sent ✅" if send_email(email, "Shortlisted: Application Update", body, s_email, s_pass) else "Failed ❌"
        else:
            e_status = "Config Missing"

        results.append({
            "Filename": file.name, 
            "Email": email if email else "N/A", 
            "Score": score, 
            "Status": status, 
            "Email Status": e_status, 
            "Missing Skills": ", ".join(missing)
        })

    # Show Results
    df = pd.DataFrame(results)
    st.divider()
    st.subheader("Analysis Summary")
    
    if not df.empty:
        def color_status(val):
            color = '#d4edda' if val == 'SELECTED' else '#f8d7da'
            return f'background-color: {color}'

        try:
            styled_df = df.style.map(color_status, subset=['Status'])
        except AttributeError:
            styled_df = df.style.applymap(color_status, subset=['Status'])
            
        st.dataframe(styled_df, use_container_width=True)
    
    save_to_db(results)
    st.success("Hybrid Analysis Complete! Data saved to local history.")
