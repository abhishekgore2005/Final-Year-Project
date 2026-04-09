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
st.set_page_config(page_title="AI Resume Intelligence Pro", layout="wide")

# --- 2. DATABASE SETUP ---
def init_db():
    # Changed filename to ensure a clean table structure with the new columns
    conn = sqlite3.connect('resume_intelligence.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candidates
                 (date TEXT, filename TEXT, email TEXT, phone TEXT, score REAL, status TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(data_list):
    conn = sqlite3.connect('resume_intelligence.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in data_list:
        # We use the exact keys defined in the results dictionary below
        c.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)", 
                  (timestamp, row['Filename'], row['Contact Email'], row['Phone'], row['Score (%)'], row['Status']))
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
    if not resume_text: return 0.0, required_skills, []
    
    # NLP Component
    jd = " ".join(required_skills)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd])
        nlp_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    except: nlp_score = 0.0

    # Keyword Component
    found_keywords = [s for s in required_skills if s.lower() in resume_text]
    keyword_score = (len(found_keywords) / len(required_skills)) * 100 if required_skills else 0
    
    final_score = round((nlp_score * 0.5) + (keyword_score * 0.5), 1)
    missing = [s for s in required_skills if s.lower() not in resume_text]
    return final_score, missing, found_keywords

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
st.title("🚀 AI Resume Intelligence & Recruitment Suite")

with st.sidebar:
    st.header("⚙️ Configuration")
    req_skills_input = st.text_area("Target Skills", "python, sql, machine learning")
    cutoff = st.slider("Success Threshold (%)", 0, 100, 40)
    
    st.divider()
    st.header("📧 Automation Settings")
    enable_email = st.checkbox("Enable Shortlist Alerts")
    s_email = st.text_input("Sender Email", value="hirebot.project@gmail.com")
    s_pass = st.text_input("App Password", type="password", value="nfyq ghye qzlw bmcb")

uploaded_files = st.file_uploader("Batch Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Run Intelligence Engine"):
    results = []
    skills_list = [s.strip().lower() for s in req_skills_input.split(",")]
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        
        # Robust Regex for Email and Phone
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        
        email = email_match.group(0) if email_match else "N/A"
        phone = phone_match.group(0) if phone_match else "N/A"
        
        score, missing, found = calculate_hybrid_score(text, skills_list)
        status = "SELECTED" if score >= cutoff else "REJECTED"
        
        # Fit Label Logic
        if score >= 75: fit = "🔥 High"
        elif score >= 40: fit = "✅ Moderate"
        else: fit = "❌ Low"

        # Email Notification Logic
        if not enable_email:
            e_status = "Disabled"
        elif status == "REJECTED":
            e_status = "Skipped"
        elif email == "N/A":
            e_status = "No Email Found"
        else:
            body = f"<h3>Congratulations!</h3><p>Your resume scored {score}%. We will contact you soon for an interview.</p>"
            e_status = "Sent ✅" if send_email(email, "Shortlisted!", body, s_email, s_pass) else "Failed ❌"

        # Dictionary keys must match what save_to_db calls!
        results.append({
            "Filename": file.name,
            "Contact Email": email,
            "Phone": phone,
            "Score (%)": score,
            "Fit Level": fit,
            "Status": status,
            "Email Status": e_status,
            "Missing Skills": ", ".join(missing),
            "Matched Skills": ", ".join(found)
        })

    df = pd.DataFrame(results)
    st.divider()
    
    # Custom Styling
    def color_status(val):
        if val == 'SELECTED': return 'background-color: #d4edda; color: #155724'
        if val == 'REJECTED': return 'background-color: #f8d7da; color: #721c24'
        return ''

    try:
        styled_df = df.style.map(color_status, subset=['Status'])
    except AttributeError:
        styled_df = df.style.applymap(color_status, subset=['Status'])
        
    st.subheader("📋 Candidate Intelligence Report")
    st.dataframe(styled_df, use_container_width=True)
    
    # Export functionality
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Analysis to CSV",
        data=csv,
        file_name=f"recruitment_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    
    # Final database save
    save_to_db(results)
    st.success(f"Analysis Complete! Processed {len(uploaded_files)} candidate(s).")
