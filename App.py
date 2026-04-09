import streamlit as st
import pypdf
import smtplib
import re
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit.components.v1 as components

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Screener Pro", layout="wide")

# --- 2. GOOGLE SHEETS INTEGRATION ---
def save_to_gsheet(data_list):
    try:
        # Define the scope
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Load credentials from Streamlit Secrets
        # In Streamlit Cloud: Settings > Secrets > Paste your JSON here
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open your sheet by name
        sheet = client.open("Resume_Data_History").sheet1 
        
        for row in data_list:
            # Append row: [Date, Filename, Email, Score, Status, Missing Skills]
            new_row = [
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                row['Filename'], row['Email'], row['Score'], 
                row['Status'], row['Missing Skills']
            ]
            sheet.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"GSheet Error: {e}")
        return False

# --- 3. CORE LOGIC ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = pypdf.PdfReader(file)
        return "".join([page.extract_text() or "" for page in pdf_reader.pages]).lower()
    except: return ""

def calculate_score_nlp(resume_text, required_skills):
    jd = " ".join(required_skills)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd])
        score = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)
    except: score = 0.0
    missing = [s for s in required_skills if s.lower() not in resume_text]
    return score, missing

# --- 4. MAIN INTERFACE ---
st.title("🚀 AI Resume Screener (Cloud Edition)")

tab1, tab2 = st.tabs(["Analyze", "Power BI Analytics"])

with tab1:
    req_skills = st.text_area("Skills", "python, sql")
    uploaded_files = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Analyze"):
        results = []
        skills_list = [s.strip().lower() for s in req_skills.split(",")]
        
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', text)
            score, missing = calculate_score_nlp(text, skills_list)
            status = "SELECTED" if score >= 40 else "REJECTED"
            
            results.append({
                "Filename": file.name, "Email": email_match.group(0) if email_match else None,
                "Score": score, "Status": status, "Missing Skills": ", ".join(missing)
            })
        
        st.dataframe(pd.DataFrame(results))
        # Trigger Cloud Save
        if save_to_gsheet(results):
            st.success("Data synced to Google Sheets for Power BI!")

with tab2:
    # Replace with your actual Published Embed Link
    pbi_url = "https://app.powerbi.com/view?r=YOUR_EMBED_LINK"
    components.iframe(pbi_url, height=700)
