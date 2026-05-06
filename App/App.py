# Developed by dnoobnerd [https://dnoobnerd.netlify.app]    Made with Streamlit


###### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import hashlib
import secrets
import re
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
from ml_role_model import predict_job_role
from matching_scoring import resume_jd_match_percentage, improved_resume_score
import nltk
nltk.download('stopwords')


###### Preprocessing functions ######


def _looks_like_person_name(value):
    if not value or not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) < 3 or len(v) > 60:
        return False
    if any(ch.isdigit() for ch in v):
        return False
    bad_tokens = ("skill", "programming", "html", "css", "javascript", "react", "node", "python", "java", "android")
    low = v.lower()
    if any(t in low for t in bad_tokens):
        return False
    if v.isupper() and " " not in v:
        return False
    return True


def _extract_emails_from_text(text):
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)


def _normalize_parsed_resume_fields(resume_data, resume_text, form_name, form_email, form_mobile):
    """Fix common pyresparser PDF layout issues (phone glued to email, skills mistaken as name)."""
    if not resume_data:
        return resume_data

    name = resume_data.get("name")
    email = resume_data.get("email")
    mobile = resume_data.get("mobile_number")

    # Email: strip leading phone digits accidentally merged
    if email and isinstance(email, str):
        m = re.match(r"^(\d{10,15})([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})$", email.strip())
        if m:
            if not mobile:
                mobile = m.group(1)
            email = m.group(2)

    # Email: try regex from raw text if still invalid-looking
    if email and "@" not in str(email):
        emails = _extract_emails_from_text(resume_text)
        if emails:
            email = emails[0]

    # Name: prefer form name if parser output is garbage
    if form_name and not _looks_like_person_name(name):
        name = form_name.strip()

    # Email: prefer form email if parser still odd
    if form_email and form_email.strip():
        fe = form_email.strip()
        if not email or "@" not in str(email) or re.search(r"\d{10,}", str(email)):
            email = fe

    # Mobile: prefer form mobile if missing
    if form_mobile and not mobile:
        mobile = form_mobile.strip()

    resume_data["name"] = name
    resume_data["email"] = email
    resume_data["mobile_number"] = mobile
    return resume_data


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######


# sql connector (override with env vars if needed)
connection = pymysql.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    user=os.getenv('DB_USER', 'root'),
    password=os.getenv('DB_PASSWORD', ''),
    db=os.getenv('DB_NAME', 'cv')
)
cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(user_id,sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    (ID,user_id,sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,Name,Email_ID,resume_score,Timestamp,Page_no,Predicted_Field,User_level,Actual_skills,Recommended_skills,Recommended_courses,pdf_name)
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (user_id,str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_admin_user(username, password):
    insert_sql = "INSERT INTO admin_users (username, password_hash) VALUES (%s, %s)"
    cursor.execute(insert_sql, (username.strip(), hash_password(password)))
    connection.commit()


def verify_admin_user(username, password):
    query = "SELECT password_hash FROM admin_users WHERE username=%s"
    cursor.execute(query, (username.strip(),))
    result = cursor.fetchone()
    if not result:
        return False
    return result[0] == hash_password(password)


def admin_user_count():
    cursor.execute("SELECT COUNT(*) FROM admin_users")
    result = cursor.fetchone()
    return int(result[0]) if result else 0


def create_user_account(name, email, password):
    insert_sql = "INSERT INTO user_accounts (name, email, password_hash) VALUES (%s, %s, %s)"
    cursor.execute(insert_sql, (name.strip(), email.strip().lower(), hash_password(password)))
    connection.commit()


def verify_user_credentials(email, password):
    query = "SELECT id, password_hash FROM user_accounts WHERE email=%s"
    cursor.execute(query, (email.strip().lower(),))
    row = cursor.fetchone()
    if not row:
        return None
    user_id, stored_hash = row
    return int(user_id) if stored_hash == hash_password(password) else None


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='./Logo/recommend.png',
   layout="wide",
)


###### Main function run() ######
def apply_ui_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top right, #dbeafe 0%, #f8fafc 35%, #eef2ff 100%);
                color: #0f172a !important;
            }
            .block-container {
                max-width: 1200px;
                padding-top: 1.2rem;
                padding-bottom: 2.5rem;
            }
            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.1rem;
            }
            .subtitle {
                color: #475569;
                margin-bottom: 0.9rem;
            }
            .section-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #dbe4f0;
                border-radius: 16px;
                padding: 18px 18px 10px 18px;
                margin-bottom: 14px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }
            .stButton > button {
                border-radius: 12px;
                border: 1px solid #2563eb;
                background: linear-gradient(90deg, #2563eb, #1d4ed8);
                color: white;
                font-weight: 600;
            }
            .stTextInput > div > div > input {
                border-radius: 10px;
                border: 1px solid #cbd5e1 !important;
                background: #ffffff !important;
                color: #0f172a !important;
            }
            .stTextArea textarea {
                border-radius: 10px;
                border: 1px solid #cbd5e1 !important;
                background: #ffffff !important;
                color: #0f172a !important;
            }
            .stTextInput label, .stTextArea label, .stSelectbox label, .stRadio label {
                color: #0f172a !important;
                font-weight: 600;
            }
            .stSelectbox > div > div {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
            }
            /* Selectbox dropdown menu (options) readability fix */
            div[data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0f172a !important;
            }
            div[data-baseweb="popover"] ul {
                background: #ffffff !important;
            }
            div[data-baseweb="popover"] li {
                background: #ffffff !important;
                color: #0f172a !important;
            }
            div[data-baseweb="popover"] li:hover {
                background: #e2e8f0 !important;
                color: #0f172a !important;
            }
            [data-testid="stForm"] {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px;
            }
            .stFileUploader {
                border-radius: 12px;
                background: #f8fafc;
                border: 1px dashed #93c5fd;
                padding: 8px;
            }
            .small-note {
                color: #64748b;
                font-size: 0.92rem;
            }
            p, li, span, label, h1, h2, h3, h4, h5, h6, div {
                color: #0f172a;
            }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stCaptionContainer"] {
                color: #0f172a !important;
            }
            /* Metric typography tuning (fix oversized values) */
            [data-testid="stMetricLabel"] {
                font-size: 0.82rem !important;
                color: #475569 !important;
                font-weight: 600 !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.05rem !important;
                font-weight: 700 !important;
                line-height: 1.25 !important;
                color: #0f172a !important;
            }
            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 10px 12px;
                min-height: 76px;
            }
            [data-testid="stAlert"] {
                color: #0f172a !important;
            }
            .stSuccess, .stInfo, .stWarning, .stError {
                color: #0f172a !important;
            }
            section[data-testid="stSidebar"] > div {
                background: #0f172a;
                color: #e2e8f0;
            }
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {
                color: #e2e8f0 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stRadio"] label {
                padding: 0.28rem 0.4rem;
                border-radius: 8px;
                margin-bottom: 0.1rem;
            }
            section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
                background: #1e293b;
                border-left: 3px solid #3b82f6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header():
    img = Image.open('./Logo/RESUM.png')
    col_logo, col_head = st.columns([1, 4])
    with col_logo:
        st.image(img, width=140)
    with col_head:
        st.markdown("<div class='main-title'>AI Resume Analyzer</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Analyze resumes, predict roles, match against JD, and review insights from a clean dashboard.</div>",
            unsafe_allow_html=True,
        )


def render_sidebar_nav():
    st.sidebar.markdown("### Navigation")
    nav_items = [
        "📊 Analysis",
        "📁 My Reports",
        "🛠️ Admin",
        "💬 Feedback",
        "ℹ️ About",
    ]
    # Restore selected page from query/session so refresh keeps the same section
    default_page = "Analysis"
    if hasattr(st, "query_params"):
        default_page = str(st.query_params.get("page", default_page))
    else:
        qp = st.experimental_get_query_params()
        default_page = qp.get("page", [default_page])[0]

    if "menu_page" not in st.session_state:
        st.session_state["menu_page"] = default_page

    page_to_label = {
        "Analysis": "📊 Analysis",
        "My Reports": "📁 My Reports",
        "Admin": "🛠️ Admin",
        "Feedback": "💬 Feedback",
        "About": "ℹ️ About",
    }
    default_label = page_to_label.get(st.session_state["menu_page"], "📊 Analysis")
    default_index = nav_items.index(default_label) if default_label in nav_items else 0

    selected_label = st.sidebar.radio("Menu", nav_items, index=default_index)
    selected_page = selected_label.split(" ", 1)[1]
    st.session_state["menu_page"] = selected_page

    if hasattr(st, "query_params"):
        st.query_params["page"] = selected_page
    else:
        qp = st.experimental_get_query_params()
        qp["page"] = selected_page
        st.experimental_set_query_params(**qp)

    return selected_page


def open_card():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def safe_divider():
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")


def get_auth_from_query():
    if hasattr(st, "query_params"):
        qp = st.query_params
        admin_flag = qp.get("admin", "0")
        username = qp.get("admin_user", "")
        return str(admin_flag) == "1", str(username)
    qp = st.experimental_get_query_params()
    admin_flag = qp.get("admin", ["0"])[0]
    username = qp.get("admin_user", [""])[0]
    return str(admin_flag) == "1", str(username)


def set_auth_query(logged_in, username=""):
    current_page = st.session_state.get("menu_page", "Analysis")
    if hasattr(st, "query_params"):
        qp = st.query_params
        if logged_in:
            qp["admin"] = "1"
            qp["admin_user"] = username
            qp["page"] = "Admin"
        else:
            qp.clear()
            qp["page"] = current_page
        return
    if logged_in:
        st.experimental_set_query_params(admin="1", admin_user=username, page="Admin")
    else:
        st.experimental_set_query_params(page=current_page)


def render_result_cards(predicted_role, ml_confidence, improved_score_value, jd_match_pct=None):
    col1, col2, col3 = st.columns(3)
    with col1:
        open_card()
        st.subheader("Prediction")
        st.metric("Job Role", predicted_role)
        st.caption(f"Confidence: {ml_confidence*100:.1f}%")
        close_card()
    with col2:
        open_card()
        st.subheader("Resume Score")
        st.metric("Score", f"{improved_score_value}/100")
        st.progress(min(max(int(improved_score_value), 0), 100))
        close_card()
    with col3:
        open_card()
        st.subheader("JD Match")
        if jd_match_pct is None:
            st.info("Add job description to calculate match.")
        else:
            st.metric("Match %", f"{jd_match_pct}%")
            st.progress(min(max(int(jd_match_pct), 0), 100))
        close_card()


def render_skill_pills(title, skills):
    st.markdown(f"#### {title}")
    if not skills:
        st.info("No skills found.")
        return
    pills = []
    for skill in skills:
        pills.append(
            f"<span style='display:inline-block;background:#eef2ff;color:#3730a3;"
            f"border:1px solid #c7d2fe;border-radius:999px;padding:4px 10px;"
            f"margin:4px 6px 4px 0;font-size:12px;font-weight:600;'>{skill}</span>"
        )
    st.markdown("".join(pills), unsafe_allow_html=True)


def generate_resume_suggestions(resume_text, predicted_role, extracted_skills):
    text = (resume_text or "").lower()
    skills_set = {s.lower() for s in (extracted_skills or [])}
    suggestions = []

    # Missing sections checks
    required_sections = {
        "Projects": ["project", "projects"],
        "Certifications": ["certification", "certifications"],
        "Experience": ["experience", "work experience", "internship", "internships"],
        "Education": ["education", "degree", "university", "college"],
        "Skills": ["skills", "technical skills"],
        "Summary/Objective": ["summary", "objective", "profile"],
    }
    for section_name, keywords in required_sections.items():
        if not any(k in text for k in keywords):
            suggestions.append(f"Add a clear **{section_name}** section to strengthen your resume.")

    # Missing role-based skills checks
    role_skill_map = {
        "Data Science": ["python", "sql", "pandas", "numpy", "machine learning"],
        "Web Development": ["html", "css", "javascript", "react", "node"],
        "Android Development": ["android", "kotlin", "java", "xml", "firebase"],
        "IOS Development": ["ios", "swift", "xcode", "objective-c"],
        "UI-UX Development": ["figma", "wireframe", "prototype", "user research"],
    }
    target_skills = role_skill_map.get(predicted_role, [])
    missing_skills = [s for s in target_skills if s not in skills_set and s not in text]
    if missing_skills:
        suggestions.append(
            f"For **{predicted_role}**, consider adding skills like: {', '.join(missing_skills[:5])}."
        )

    if not suggestions:
        suggestions.append("Great work! Your resume already includes most important sections and role-relevant skills.")

    return suggestions


def build_pdf_report(candidate_name, predicted_role, score, match_pct, suggestions):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Resume Analysis Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Name: {candidate_name or 'NA'}")
    y -= 20
    pdf.drawString(50, y, f"Predicted Role: {predicted_role}")
    y -= 20
    pdf.drawString(50, y, f"Resume Score: {score}/100")
    y -= 20
    match_text = f"{match_pct}%" if match_pct is not None else "NA"
    pdf.drawString(50, y, f"JD Match Percentage: {match_text}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Improvement Suggestions:")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for s in suggestions:
        line = f"- {s}"
        if y < 60:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)
        pdf.drawString(60, y, line[:130])
        y -= 18

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def skill_gap_analysis(predicted_role, extracted_skills, resume_text):
    role_skill_map = {
        "Data Science": {
            "important": ["python", "machine learning", "sql", "pandas", "numpy"],
            "optional": ["tensorflow", "pytorch", "statistics", "scikit-learn", "data visualization"],
        },
        "Web Development": {
            "important": ["html", "css", "javascript", "react", "node"],
            "optional": ["typescript", "django", "flask", "rest api", "mongodb"],
        },
        "Android Development": {
            "important": ["android", "kotlin", "java", "xml"],
            "optional": ["firebase", "flutter", "sqlite", "gradle"],
        },
        "IOS Development": {
            "important": ["ios", "swift", "xcode"],
            "optional": ["objective-c", "cocoa", "ui kit", "swiftui"],
        },
        "UI-UX Development": {
            "important": ["figma", "wireframe", "prototype", "user research"],
            "optional": ["adobe xd", "usability testing", "design system"],
        },
    }

    role_map = role_skill_map.get(predicted_role, {"important": [], "optional": []})
    skills_set = {s.lower() for s in (extracted_skills or [])}
    text = (resume_text or "").lower()

    def _missing(skill_list):
        return [s for s in skill_list if s not in skills_set and s not in text]

    missing_important = _missing(role_map["important"])
    missing_optional = _missing(role_map["optional"])
    return missing_important, missing_optional


def run():
    
    apply_ui_theme()
    render_top_header()
    choice = render_sidebar_nav()

    ###### Creating Database and Table ######


    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)


    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    user_id INT NULL,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)
    # ensure user_id exists in old databases
    try:
        cursor.execute("ALTER TABLE user_data ADD COLUMN user_id INT NULL AFTER ID")
    except Exception:
        pass


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)

    # Create admin users table for login
    admin_table_sql = """
        CREATE TABLE IF NOT EXISTS admin_users (
            id INT NOT NULL AUTO_INCREMENT,
            username VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        );
    """
    cursor.execute(admin_table_sql)

    # Create user accounts table for signup/login
    user_account_sql = """
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INT NOT NULL AUTO_INCREMENT,
            name VARCHAR(120) NOT NULL,
            email VARCHAR(180) NOT NULL UNIQUE,
            password_hash VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        );
    """
    cursor.execute(user_account_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

    def user_section():
        open_card()
        st.subheader("User Details")
        st.markdown("<p class='small-note'>Fill and save profile details for resume analysis.</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            act_name = st.text_input('Name*', value=st.session_state.get("act_name", ""), placeholder="Enter your full name")
        with col2:
            act_mail = st.text_input('Mail*', value=st.session_state.get("act_mail", ""), placeholder="you@example.com")
        with col3:
            act_mob  = st.text_input('Mobile Number*', value=st.session_state.get("act_mob", ""), placeholder="+91 xxxxxxxxxx")
        if st.button("Save Details"):
            st.session_state["act_name"] = act_name
            st.session_state["act_mail"] = act_mail
            st.session_state["act_mob"] = act_mob
            st.success("Details saved.")
        close_card()

    def upload_section():
        open_card()
        st.subheader("Upload Resume")
        st.caption("Upload resume PDF and optional job description.")
        job_description = st.text_area(
            "Paste Job Description (optional, for JD matching)",
            value=st.session_state.get("job_description", ""),
            placeholder="Paste the target job description here to get resume-job match percentage...",
            height=140,
        )
        st.session_state["job_description"] = job_description
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        close_card()
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name_local = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            parsed_resume_data = ResumeParser(save_image_path).get_extracted_data()
            if parsed_resume_data:
                parsed_resume_text = pdf_reader(save_image_path)
                st.session_state["save_image_path"] = save_image_path
                st.session_state["pdf_name"] = pdf_name_local
                st.session_state["resume_data"] = parsed_resume_data
                st.session_state["resume_text"] = parsed_resume_text
                st.success("Resume uploaded and parsed successfully. Open Analysis section.")
            else:
                st.error("Unable to parse this resume.")

    def analysis_section():
        if "user_logged_in" not in st.session_state:
            st.session_state["user_logged_in"] = False
        if "user_id" not in st.session_state:
            st.session_state["user_id"] = None

        # lightweight user auth UI
        if not st.session_state["user_logged_in"]:
            open_card()
            st.subheader("User Login / Signup")
            auth_tab = st.selectbox("Choose action", ["Login", "Signup"])
            if auth_tab == "Login":
                with st.form("user_login_form"):
                    user_email = st.text_input("Email")
                    user_password = st.text_input("Password", type="password")
                    user_login_submit = st.form_submit_button("Login")
                    if user_login_submit:
                        user_id_val = verify_user_credentials(user_email, user_password)
                        if user_id_val:
                            st.session_state["user_logged_in"] = True
                            st.session_state["user_id"] = user_id_val
                            st.success("User login successful.")
                            if hasattr(st, "rerun"):
                                st.rerun()
                            else:
                                st.experimental_rerun()
                        else:
                            st.error("Invalid user credentials.")
            else:
                with st.form("user_signup_form"):
                    signup_name = st.text_input("Name")
                    signup_email = st.text_input("Email")
                    signup_password = st.text_input("Password", type="password")
                    signup_submit = st.form_submit_button("Create Account")
                    if signup_submit:
                        if not signup_name.strip() or not signup_email.strip() or not signup_password.strip():
                            st.warning("All signup fields are required.")
                        else:
                            try:
                                create_user_account(signup_name, signup_email, signup_password)
                                st.success("Account created. Now login.")
                            except Exception:
                                st.error("Email already exists or invalid input.")
            close_card()
            return

        top_user_left, top_user_right = st.columns([4, 1])
        with top_user_left:
            st.success(f"User logged in (ID: {st.session_state.get('user_id')})")
        with top_user_right:
            if st.button("User Logout"):
                st.session_state["user_logged_in"] = False
                st.session_state["user_id"] = None
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

        open_card()
        st.subheader("Analysis")
        st.caption("Enter your details, upload resume, then review analysis on same page.")
        c1, c2, c3 = st.columns(3)
        with c1:
            act_name = st.text_input('Name*', value=st.session_state.get("act_name", ""), placeholder="Enter your full name")
        with c2:
            act_mail = st.text_input('Mail*', value=st.session_state.get("act_mail", ""), placeholder="you@example.com")
        with c3:
            act_mob = st.text_input('Mobile Number*', value=st.session_state.get("act_mob", ""), placeholder="+91 xxxxxxxxxx")

        job_description = st.text_area(
            "Paste Job Description (optional, for JD matching)",
            value=st.session_state.get("job_description", ""),
            placeholder="Paste the target job description here to get resume-job match percentage...",
            height=120,
        )
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"], key="analysis_resume_uploader")
        close_card()

        st.session_state["act_name"] = act_name
        st.session_state["act_mail"] = act_mail
        st.session_state["act_mob"] = act_mob
        st.session_state["job_description"] = job_description

        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(2)
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            resume_text = pdf_reader(save_image_path) if resume_data else None
            st.session_state["save_image_path"] = save_image_path
            st.session_state["pdf_name"] = pdf_name
            st.session_state["resume_data"] = resume_data
            st.session_state["resume_text"] = resume_text
        else:
            resume_data = st.session_state.get("resume_data")
            resume_text = st.session_state.get("resume_text")
            save_image_path = st.session_state.get("save_image_path")
            pdf_name = st.session_state.get("pdf_name")

        if resume_data and resume_text:
            resume_data = _normalize_parsed_resume_fields(
                resume_data, resume_text, act_name, act_mail, act_mob
            )
            st.session_state["resume_data"] = resume_data

        if not resume_data or not resume_text or not save_image_path:
            st.info("Upload resume to start analysis.")
            return

        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        location = geolocator.reverse(latlong, language='en')
        address = location.raw['address']
        cityy = address.get('city', '')
        statee = address.get('state', '')
        countryy = address.get('country', '')
        city = cityy
        state = statee
        country = countryy

        if True:
                ## Showing Analyzed data from (resume_data)
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.header("**Resume Analysis 🤘**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info 👀**")
                try:
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Name", str(resume_data['name']) if resume_data['name'] else "NA")
                        st.metric("Email", str(resume_data['email']) if resume_data['email'] else "NA")
                    with col_b2:
                        st.metric("Contact", str(resume_data['mobile_number']) if resume_data['mobile_number'] else "NA")
                        st.metric("Degree", str(resume_data['degree']))
                    with col_b3:
                        st.metric("Pages", str(resume_data['no_of_pages']))
                        st.metric("Profile Email", act_mail if act_mail else "NA")

                except:
                    pass
                st.markdown("</div>", unsafe_allow_html=True)
                ## Predicting Candidate Experience Level 

                ### Trying with different possibilities
                cand_level = ''
                if resume_data['no_of_pages'] < 1:                
                    cand_level = "NA"
                    st.warning("You are at fresher level.")
                
                #### if internship then intermediate level
                elif 'INTERNSHIP' in resume_text:
                    cand_level = "Intermediate"
                    st.info("You are at intermediate level.")
                elif 'INTERNSHIPS' in resume_text:
                    cand_level = "Intermediate"
                    st.info("You are at intermediate level.")
                elif 'Internship' in resume_text:
                    cand_level = "Intermediate"
                    st.info("You are at intermediate level.")
                elif 'Internships' in resume_text:
                    cand_level = "Intermediate"
                    st.info("You are at intermediate level.")
                
                #### if Work Experience/Experience then Experience level
                elif 'EXPERIENCE' in resume_text:
                    cand_level = "Experienced"
                    st.success("You are at experience level.")
                elif 'WORK EXPERIENCE' in resume_text:
                    cand_level = "Experienced"
                    st.success("You are at experience level.")
                elif 'Experience' in resume_text:
                    cand_level = "Experienced"
                    st.success("You are at experience level.")
                elif 'Work Experience' in resume_text:
                    cand_level = "Experienced"
                    st.success("You are at experience level.")
                else:
                    cand_level = "Fresher"
                    st.warning("You are at fresher level.")


                ## Skills Analyzing and Recommendation
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("**Skills Recommendation 💡**")
                
                ### Current Analyzed Skills
                render_skill_pills("Your Current Skills", resume_data.get('skills', []))

                ### Keywords for Recommendations
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
                n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                ### Skill Recommendations Starts                
                recommended_skills = []
                reco_field = ''
                rec_course = ''

                # ML-based role prediction (TF-IDF + Logistic Regression model)
                predicted_role, role_confidence, _ = predict_job_role(resume_text)
                st.success(f"ML Prediction: {predicted_role} ({role_confidence*100:.1f}% confidence)")

                # Resume vs Job Description matching (TF-IDF + cosine similarity)
                jd_match_pct = None
                if job_description and job_description.strip():
                    jd_match_pct = resume_jd_match_percentage(resume_text, job_description)
                    st.info(f"Resume vs Job Description Match: {jd_match_pct}%")

                if predicted_role == 'Data Science':
                    reco_field = 'Data Science'
                    recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                    render_skill_pills("Recommended Skills for You", recommended_skills)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills can improve your resume relevance for Data Science roles.</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(ds_course)
                elif predicted_role == 'Web Development':
                    reco_field = 'Web Development'
                    recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                    render_skill_pills("Recommended Skills for You", recommended_skills)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills can improve your resume relevance for Web Development roles.</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(web_course)
                elif predicted_role == 'Android Development':
                    reco_field = 'Android Development'
                    recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                    render_skill_pills("Recommended Skills for You", recommended_skills)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills can improve your resume relevance for Android roles.</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(android_course)
                elif predicted_role == 'IOS Development':
                    reco_field = 'IOS Development'
                    recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                    render_skill_pills("Recommended Skills for You", recommended_skills)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills can improve your resume relevance for IOS roles.</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(ios_course)
                elif predicted_role == 'UI-UX Development':
                    reco_field = 'UI-UX Development'
                    recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                    render_skill_pills("Recommended Skills for You", recommended_skills)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills can improve your resume relevance for UI/UX roles.</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(uiux_course)

                ### condition starts to check skills from keywords and predict field
                for i in []:
                
                    #### Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ds_course)
                        break

                    #### Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(web_course)
                        break

                    #### Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(android_course)
                        break

                    #### IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ios_course)
                        break

                    #### Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(uiux_course)
                        break

                    #### For Not Any Recommendations
                    elif i.lower() in n_any:
                        print(i.lower())
                        reco_field = 'NA'
                        st.warning("** Currently our tool only predicts and recommends for Data Science, Web, Android, IOS and UI/UX Development**")
                        recommended_skills = ['No Recommendations']
                        render_skill_pills("Recommended Skills for You", recommended_skills)
                        st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = "Sorry! Not Available for this Field"
                        break
                st.markdown("</div>", unsafe_allow_html=True)


                ## Resume Scorer & Resume Writing Tips
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("**Resume Tips & Ideas 🥂**")
                resume_score = 0
                
                ### Predicting Whether these key points are added to the resume
                if 'Objective' or 'Summary' in resume_text:
                    resume_score = resume_score+6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective/Summary</h4>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Education' or 'School' or 'College'  in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education Details</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Education. It will give Your Qualification level to the recruiter</h4>''',unsafe_allow_html=True)

                if 'EXPERIENCE' in resume_text:
                    resume_score = resume_score + 16
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                elif 'Experience' in resume_text:
                    resume_score = resume_score + 16
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Experience. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'INTERNSHIPS'  in resume_text:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'INTERNSHIP'  in resume_text:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'Internships'  in resume_text:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'Internship'  in resume_text:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Internships. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'SKILLS'  in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'SKILL'  in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skills'  in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skill'  in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Skills. It will help you a lot</h4>''',unsafe_allow_html=True)

                if 'HOBBIES' in resume_text:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                elif 'Hobbies' in resume_text:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                if 'INTERESTS'in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                elif 'Interests'in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Interest. It will show your interest other that job.</h4>''',unsafe_allow_html=True)

                if 'ACHIEVEMENTS' in resume_text:
                    resume_score = resume_score + 13
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                elif 'Achievements' in resume_text:
                    resume_score = resume_score + 13
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                if 'CERTIFICATIONS' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certifications' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certification' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Certifications. It will show that you have done some specialization for the required position.</h4>''',unsafe_allow_html=True)

                if 'PROJECTS' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'PROJECT' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Projects' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Project' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

                st.subheader("**Resume Score 📝**")
                
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )

                ### Score Bar
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                ### Score
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")
                st.markdown("</div>", unsafe_allow_html=True)

                # Improved simple scoring (skills + length + keyword relevance)
                open_card()
                st.subheader("Improved Resume Score")
                target_keywords = {
                    'Data Science': ['python', 'machine learning', 'pandas', 'sql', 'statistics', 'tensorflow'],
                    'Web Development': ['javascript', 'react', 'node', 'html', 'css', 'api'],
                    'Android Development': ['android', 'kotlin', 'java', 'xml', 'flutter', 'firebase'],
                    'IOS Development': ['ios', 'swift', 'xcode', 'cocoa', 'objective-c'],
                    'UI-UX Development': ['figma', 'wireframe', 'prototype', 'ux', 'ui', 'research'],
                }
                selected_keywords = target_keywords.get(reco_field, ['python', 'sql', 'project', 'skills', 'experience'])
                improved_score_value, score_breakdown = improved_resume_score(
                    resume_text=resume_text,
                    extracted_skills=resume_data.get('skills', []),
                    target_keywords=selected_keywords,
                )
                st.success(f"Improved Resume Score: {improved_score_value}/100")
                st.caption(
                    f"Skills: {score_breakdown['skills_score']}/40 | "
                    f"Length: {score_breakdown['length_score']}/30 | "
                    f"Relevance: {score_breakdown['relevance_score']}/30 | "
                    f"Words: {score_breakdown['word_count']}"
                )
                close_card()
                render_result_cards(
                    predicted_role=predicted_role,
                    ml_confidence=role_confidence,
                    improved_score_value=improved_score_value,
                    jd_match_pct=jd_match_pct,
                )
                resume_score = improved_score_value

                # Resume improvement suggestions (rule-based)
                open_card()
                st.subheader("Resume Improvement Suggestions")
                improvement_suggestions = generate_resume_suggestions(
                    resume_text=resume_text,
                    predicted_role=predicted_role,
                    extracted_skills=resume_data.get("skills", []),
                )
                for item in improvement_suggestions:
                    st.markdown(f"- {item}")
                pdf_bytes = build_pdf_report(
                    candidate_name=resume_data.get("name", ""),
                    predicted_role=predicted_role,
                    score=improved_score_value,
                    match_pct=jd_match_pct,
                    suggestions=improvement_suggestions,
                )
                st.download_button(
                    label="Download Analysis PDF",
                    data=pdf_bytes,
                    file_name="resume_analysis_report.pdf",
                    mime="application/pdf",
                )
                close_card()

                # Skill Gap Analysis
                open_card()
                st.subheader("Skill Gap Analysis")
                missing_important, missing_optional = skill_gap_analysis(
                    predicted_role=predicted_role,
                    extracted_skills=resume_data.get("skills", []),
                    resume_text=resume_text,
                )
                if missing_important:
                    st.warning("Important missing skills:")
                    for s in missing_important:
                        st.markdown(f"- {s}")
                else:
                    st.success("No important skill gaps found for this role.")

                if missing_optional:
                    st.info("Optional skills to improve profile:")
                    for s in missing_optional:
                        st.markdown(f"- {s}")
                else:
                    st.info("No optional skill gaps found.")
                close_card()

                # print(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)


                ### Getting Current Date and Time
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)


                ## Calling insert_data to add all the data into user_data                
                insert_data(
                    st.session_state.get("user_id"),
                    str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country),
                    (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                    str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills),
                    str(rec_course), pdf_name
                )

                ## Recommending Resume Writing Video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                ## Recommending Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                ## On Successful Result 
                st.balloons()

    def my_reports_section():
        if "user_logged_in" not in st.session_state or not st.session_state["user_logged_in"]:
            open_card()
            st.info("Please login from Analysis section to view My Reports.")
            close_card()
            return
        open_card()
        st.subheader("My Reports")
        user_id = st.session_state.get("user_id")
        query = """
            SELECT ID, Timestamp, convert(Predicted_Field using utf8), resume_score, pdf_name
            FROM user_data
            WHERE user_id=%s
            ORDER BY ID DESC
        """
        reports_df = pd.read_sql(query, connection, params=[user_id])
        if reports_df.empty:
            st.info("No reports found for your account yet.")
        else:
            reports_df.columns = ["Report ID", "Timestamp", "Predicted Field", "Resume Score", "File Name"]
            st.dataframe(reports_df, width=1000)
        close_card()

        
    if choice == 'Analysis':
        analysis_section()
    elif choice == 'My Reports':
        my_reports_section()


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':   
        open_card()
        st.subheader("Share Your Feedback")
        st.caption("Your rating helps improve recommendation quality and UI experience.")

        # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        # Feedback Form
        with st.form("my_form"):
            c1, c2 = st.columns(2)
            with c1:
                feed_name = st.text_input('Name', placeholder="Your name")
            with c2:
                feed_email = st.text_input('Email', placeholder="you@example.com")

            st.markdown("##### Rating")
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5, 4)
            comments = st.text_area('Comments', placeholder="What can we improve?", height=100)
            Timestamp = timestamp        
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not feed_name.strip() or not feed_email.strip():
                    st.warning("Please enter both name and email.")
                    close_card()
                    return
                ## Calling insertf_data to add dat into user feedback
                insertf_data(feed_name,feed_email,feed_score,comments,Timestamp)    
                ## Success Message 
                st.success("Thanks! Your Feedback was recorded.") 
                ## On Successful Submit
                st.balloons()    
        close_card()


        # query to fetch data from user feedback table
        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        # fetching feed_score from the query and getting the unique values and total value count 
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()

        if plotfeed_data.empty:
            open_card()
            st.info("No feedback submitted yet. Be the first to rate the app.")
            close_card()
            return

        col_r1, col_r2 = st.columns([1.1, 1.3])
        with col_r1:
            open_card()
            st.subheader("Past User Ratings")
            fig = px.pie(values=values, names=labels, title="Rating Distribution", color_discrete_sequence=px.colors.sequential.Aggrnyl)
            fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            close_card()

        #  Fetching Comment History
        cursor.execute('select feed_name, comments from user_feedback')
        plfeed_cmt_data = cursor.fetchall()
        with col_r2:
            open_card()
            st.subheader("User Comments")
            dff = pd.DataFrame(plfeed_cmt_data, columns=['User', 'Comment'])
            st.dataframe(dff, width=1000)
            close_card()

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':   
        open_card()
        st.subheader("About AI Resume Analyzer")
        st.write(
            "AI Resume Analyzer is a lightweight Streamlit application that helps candidates improve resumes and helps recruiters quickly evaluate profile quality."
        )
        st.markdown("### What This App Does")
        st.markdown(
            """
            - Parses resume PDF and extracts key information
            - Predicts likely job role using ML
            - Computes Resume vs Job Description match percentage
            - Generates skill/course recommendations
            - Provides resume score with simple actionable insights
            - Offers admin dashboard for user and feedback analytics
            """
        )
        st.markdown("### How To Use")
        st.markdown(
            """
            1. Go to **User** section and fill basic details.
            2. Upload your resume PDF in **Upload** section.
            3. Optionally add job description for match percentage.
            4. Open **Analysis** to view prediction, score, and recommendations.
            5. Submit your experience in **Feedback**.
            """
        )
        st.success("Tip: Keep clear sections like Skills, Projects, Experience, and Certifications in your resume.")
        close_card()


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    else:
        with st.container():
            st.subheader("Admin Dashboard")
            st.caption("Secure admin access for user reports, feedback insights, and analytics.")
            safe_divider()
        if "admin_logged_in" not in st.session_state:
            st.session_state.admin_logged_in = False
        if "admin_username" not in st.session_state:
            st.session_state.admin_username = ""

        # Restore admin auth from URL query params on refresh
        auth_from_query, query_user = get_auth_from_query()
        if auth_from_query and not st.session_state.admin_logged_in:
            st.session_state.admin_logged_in = True
            st.session_state.admin_username = query_user or "admin"

        total_admin_users = admin_user_count()

        if total_admin_users == 0:
            with st.container():
                c_left, c_center, c_right = st.columns([1, 2, 1])
                with c_center:
                    open_card()
                    st.info("No admin user found. Create your first admin account.")
                    with st.form("create_first_admin"):
                        st.markdown("### Create First Admin")
                        st.caption("Set username and password to unlock admin dashboard.")
                        new_admin_user = st.text_input("New Admin Username")
                        new_admin_password = st.text_input("New Admin Password", type="password")
                        create_admin_submit = st.form_submit_button("Create Admin User")
                        if create_admin_submit:
                            if not new_admin_user.strip() or not new_admin_password.strip():
                                st.error("Username and password are required.")
                            else:
                                try:
                                    create_admin_user(new_admin_user, new_admin_password)
                                    st.success("Admin user created. Login using new credentials.")
                                except Exception:
                                    st.error("Unable to create admin user. Try different username.")
                    close_card()
            return

        # Admin Login
        if not st.session_state.admin_logged_in:
            with st.container():
                c_left, c_center, c_right = st.columns([1, 2, 1])
                with c_center:
                    open_card()
                    st.markdown("### Admin Login")
                    st.caption("Use your admin credentials to continue.")
                    with st.form("admin_login_form"):
                        ad_user = st.text_input("Username")
                        ad_password = st.text_input("Password", type='password')
                        admin_login = st.form_submit_button('🔐 Login to Dashboard')
                        if admin_login:
                            if verify_admin_user(ad_user, ad_password):
                                st.session_state.admin_logged_in = True
                                st.session_state.admin_username = ad_user.strip()
                                set_auth_query(True, ad_user.strip())
                                st.success("Login successful.")
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:
                                    st.experimental_rerun()
                            else:
                                st.error("Wrong username or password.")
                    close_card()
            return

        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.success(f"Logged in as: {st.session_state.admin_username}")
        with top_right:
            if st.button("Logout"):
                st.session_state.pop("admin_logged_in", None)
                st.session_state.pop("admin_username", None)
                set_auth_query(False)
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
        st.markdown("### Overview")

        ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
        cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
        datanalys = cursor.fetchall()
        plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country'])
        
        ### Total Users Count with a Welcome Message
        values = plot_data.Idt.count()
        st.success("Welcome Admin. Total %d users have used the tool." % values)
        
        ### Fetch user data from user_data(table) and convert it into dataframe
        cursor.execute('''SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, resume_score, Page_no, pdf_name, convert(User_level using utf8), convert(Actual_skills using utf8), convert(Recommended_skills using utf8), convert(Recommended_courses using utf8), city, state, country, latlong, os_name_ver, host_name, dev_user from user_data''')
        data = cursor.fetchall()                

        df = pd.DataFrame(data, columns=['ID', 'Token', 'IP Address', 'Name', 'Mail', 'Mobile Number', 'Predicted Field', 'Timestamp',
                                            'Predicted Name', 'Predicted Mail', 'Resume Score', 'Total Page',  'File Name',   
                                            'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course',
                                            'City', 'State', 'Country', 'Lat Long', 'Server OS', 'Server Name', 'Server User',])

        ### Fetch feedback data from user_feedback(table) and convert it into dataframe
        cursor.execute('''SELECT * from user_feedback''')
        data = cursor.fetchall()
        feedback_df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])

        ### query to fetch data from user_feedback(table)
        query = 'select * from user_feedback'
        plotfeed_data = pd.read_sql(query, connection)                        

        # Admin summary cards
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            open_card()
            st.metric("Total Users", int(len(df)))
            close_card()
        with col_k2:
            open_card()
            st.metric("Feedback Entries", int(len(feedback_df)))
            close_card()
        with col_k3:
            open_card()
            avg_rating = float(plotfeed_data["feed_score"].astype(float).mean()) if not plotfeed_data.empty else 0.0
            st.metric("Average Rating", f"{avg_rating:.2f}/5")
            close_card()
        st.markdown("### Data Explorer")

        # Cleaner section navigation (more readable than compact tabs on older Streamlit)
        admin_view = st.selectbox("Choose section", ["User Data", "Feedback Data", "Analytics"])

        if admin_view == "User Data":
            open_card()
            st.header("User Data")
            user_preview_cols = ['ID', 'Name', 'Mail', 'Mobile Number', 'Predicted Field', 'User Level', 'Resume Score', 'Timestamp']
            user_preview = df[user_preview_cols] if not df.empty else df
            st.dataframe(user_preview, width=1100, height=260)
            with st.expander("Show full user dataset"):
                st.dataframe(df, width=1100, height=360)
            st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
            close_card()

        elif admin_view == "Feedback Data":
            open_card()
            st.header("User Feedback Data")
            st.dataframe(feedback_df, width=1100, height=320)
            close_card()

        else:
            open_card()
            st.header("Analytics")
            c1, c2 = st.columns(2)

            with c1:
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                st.subheader("User Ratings")
                fig = px.pie(values=values, names=labels, title="User Rating Score (1 - 5)", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()
                st.subheader("Predicted Fields")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field by Skills', color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.User_Level.unique()
                values = plot_data.User_Level.value_counts()
                st.subheader("Experience Level")
                fig = px.pie(df, values=values, names=labels, title="Users by Experience Level", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.resume_score.unique()
                values = plot_data.resume_score.value_counts()
                st.subheader("Resume Score Distribution")
                fig = px.pie(df, values=values, names=labels, title='Resume Score', color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                labels = plot_data.IP_add.unique()
                values = plot_data.IP_add.value_counts()
                st.subheader("Usage by IP")
                fig = px.pie(df, values=values, names=labels, title='Usage by IP Address', color_discrete_sequence=px.colors.sequential.matter_r)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.City.unique()
                values = plot_data.City.value_counts()
                st.subheader("Usage by City")
                fig = px.pie(df, values=values, names=labels, title='Usage by City', color_discrete_sequence=px.colors.sequential.Jet)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.State.unique()
                values = plot_data.State.value_counts()
                st.subheader("Usage by State")
                fig = px.pie(df, values=values, names=labels, title='Usage by State', color_discrete_sequence=px.colors.sequential.PuBu_r)
                st.plotly_chart(fig, use_container_width=True)

                labels = plot_data.Country.unique()
                values = plot_data.Country.value_counts()
                st.subheader("Usage by Country")
                fig = px.pie(df, values=values, names=labels, title='Usage by Country', color_discrete_sequence=px.colors.sequential.Purpor_r)
                st.plotly_chart(fig, use_container_width=True)
            close_card()

# Calling the main (run()) function to make the whole process run
run()
