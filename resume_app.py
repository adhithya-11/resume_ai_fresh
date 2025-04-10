import streamlit as st
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Keywords for role prediction
roles_keywords = {
    "data analyst": ["excel", "sql", "tableau", "power bi", "analysis", "statistics"],
    "data scientist": ["python", "machine learning", "scikit-learn", "model", "regression", "classification"],
    "business analyst": ["business", "requirement", "stakeholder", "report", "power bi"],
    "ml engineer": ["tensorflow", "pytorch", "deep learning", "deployment", "model", "neural"],
    "java developer": ["java", "spring", "hibernate", "microservices"],
    "devops engineer": ["docker", "kubernetes", "ci/cd", "aws", "jenkins"],
    "support engineer": ["ticket", "troubleshoot", "support", "customer", "issue"],
    "mechanical engineer": ["autocad", "solidworks", "design", "manufacturing"]
}

def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_sections(text):
    # Basic heuristic extraction
    return {
        "skills": text,
        "experience": text,
        "education": text
    }

def predict_job_role(sections):
    skill_text = json.dumps(sections.get("skills", "")).lower()
    experience_text = sections.get("experience", "").lower()
    combined_text = skill_text + " " + experience_text

    scores = {}
    for role, keywords in roles_keywords.items():
        count = sum(keyword in combined_text for keyword in keywords)
        scores[role] = count

    return max(scores, key=scores.get)

def compare_with_job_description(resume_text, job_description):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    return round(similarity * 100, 2)

# Streamlit UI
st.title("🧠 Resume AI Screener")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description (Optional)", height=200)

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    sections = extract_sections(resume_text)
    predicted_role = predict_job_role(sections)
    st.success(f"🎯 Predicted Job Role: {predicted_role.capitalize()}")

    if job_description:
        score = compare_with_job_description(resume_text, job_description)
        st.info(f"📊 Similarity with Job Description: {score}%")
