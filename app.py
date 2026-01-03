import streamlit as st
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# ------------------ NLTK Setup ------------------
nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# ------------------ Page Setup ------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🚀 Advanced AI Resume Analyzer")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------ Utility Functions ------------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join(p.text for p in doc.paragraphs)

def extract_skills(text):
    skills_db = [
        "python", "java", "sql", "machine learning", "deep learning",
        "nlp", "data analysis", "streamlit", "docker", "aws",
        "tensorflow", "pytorch", "excel", "power bi", "git",
        "linux", "rest api", "cloud"
    ]
    return sorted({skill for skill in skills_db if skill in text})

def semantic_similarity(resume, jd):
    emb = model.encode([resume, jd])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def get_keyword_recommendations(resume_text, jd_text, top_n=10):
    resume_words = set(word_tokenize(resume_text))
    jd_words = word_tokenize(jd_text)

    jd_freq = Counter(jd_words)
    keywords = [
        word for word, freq in jd_freq.most_common()
        if word not in resume_words and len(word) > 3
    ]
    return keywords[:top_n]

def resume_suggestions(score, missing_skills):
    suggestions = []

    if score < 50:
        suggestions.append("Your resume has low alignment with the job description.")
    if missing_skills:
        suggestions.append("Add missing technical skills relevant to the job.")
    suggestions.append("Use job-specific keywords in experience and projects.")
    suggestions.append("Quantify achievements using numbers and impact.")
    suggestions.append("Ensure resume format is ATS-friendly.")

    return suggestions

# ------------------ Streamlit UI ------------------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

with col2:
    jd_input = st.text_area("📝 Paste Job Description")

# ------------------ Processing ------------------

if resume_file and jd_input:
    if resume_file.type == "application/pdf":
        resume_text_raw = extract_text_from_pdf(resume_file)
    else:
        resume_text_raw = extract_text_from_docx(resume_file)

    resume_text = clean_text(resume_text_raw)
    jd_text = clean_text(jd_input)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = set(resume_skills) & set(jd_skills)
    missing_skills = set(jd_skills) - set(resume_skills)

    similarity_score = semantic_similarity(resume_text, jd_text)
    skill_score = len(matched_skills) / max(len(jd_skills), 1)
    final_score = (0.6 * similarity_score + 0.4 * skill_score) * 100

    # ------------------ Results ------------------
    st.subheader("📊 Match Results")
    st.metric("AI Match Score", f"{final_score:.2f}%")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("✅ Matched Skills")
        st.write(list(matched_skills) if matched_skills else "None")

    with col4:
        st.subheader("❌ Missing Skills")
        st.write(list(missing_skills) if missing_skills else "None")

   
    # ------------------ Keyword Recommendations ------------------
    st.subheader("🧠 JD-Based Keyword Recommendations")
    keywords = get_keyword_recommendations(resume_text, jd_text)

    if keywords:
        st.write(keywords)
    else:
        st.write("Your resume already covers most JD keywords 🎉")

    # ------------------ Resume Improvement Suggestions ------------------
    st.subheader("📄 Resume Improvement Suggestions")
    suggestions = resume_suggestions(final_score, missing_skills)

    for i, suggestion in enumerate(suggestions, 1):
        st.write(f"{i}. {suggestion}")

    # ------------------ Final Verdict ------------------
    if final_score >= 75:
        st.success("Excellent match for this role 🎯")
    elif final_score >= 50:
        st.warning("Good match – improvements recommended ⚠️")
    else:
        st.error("Low match – resume optimization needed ❌")
