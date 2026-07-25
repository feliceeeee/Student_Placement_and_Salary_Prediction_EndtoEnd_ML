import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

st.set_page_config(page_title="Student Placement (API)", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

st.sidebar.title("Info")
st.sidebar.write("Frontend Streamlit (Client) → FastAPI (Backend)")

if st.sidebar.button("Reset"):
    st.rerun()

st.title("Student Placement via API")
st.caption("Data dikirim ke FastAPI, lalu hasil ditampilkan di sini.")

with st.form("form"):
    st.markdown("### Academic")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.slider("SSC (%)", 50, 100, 70)
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, step=0.1)
    with col2:
        hsc = st.slider("HSC (%)", 50, 100, 70)
        degree = st.slider("Degree (%)", 50, 100, 70)
        entrance = st.slider("Entrance Score", 40, 100, 70)

    st.markdown("### Skills")
    col3, col4 = st.columns(2)
    with col3:
        tech = st.slider("Technical Skill", 40, 100, 70)
        internship = st.slider("Internship", 0, 5, 1)
    with col4:
        soft = st.slider("Soft Skill", 40, 100, 70)
        projects = st.slider("Projects", 0, 5, 1)

    st.markdown("### Experience")
    col5, col6 = st.columns(2)
    with col5:
        experience = st.slider("Experience (Months)", 0, 24, 6)
        cert = st.slider("Certifications", 0, 5, 1)
        extra = st.selectbox("Extracurricular", ["Yes", "No"])
    with col6:
        attendance = st.slider("Attendance (%)", 60, 100, 80)
        backlogs = st.slider("Backlogs", 0, 5, 0)

    submit = st.form_submit_button("Predict via API")

if not submit:
    st.info("Isi form lalu klik Predict")
    st.stop()

payload = {
    "gender": gender,
    "ssc_percentage": ssc,
    "hsc_percentage": hsc,
    "degree_percentage": degree,
    "cgpa": cgpa,
    "entrance_exam_score": entrance,
    "technical_skill_score": tech,
    "soft_skill_score": soft,
    "internship_count": internship,
    "live_projects": projects,
    "work_experience_months": experience,
    "certifications": cert,
    "attendance_percentage": attendance,
    "backlogs": backlogs,
    "extracurricular_activities": extra
}

try:
    response = requests.post(API_URL, json=payload)
    result = response.json()
except Exception as e:
    st.error("Gagal connect ke FastAPI")
    st.code(str(e))
    st.stop()

st.divider()

st.subheader("Your Key Features")
labels = ["CGPA", "Tech Skill", "Soft Skill", "Backlogs"]
values = [cgpa, tech, soft, backlogs]
fig, ax = plt.subplots(figsize=(6,3))
ax.bar(labels, values)
ax.set_title("Your Profile")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Prediction Result")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Placement", "Placed" if result["placement"] == 1 else "Not Placed")
with col2:
    st.metric("Confidence", f"{result['probability']:.2%}")
    
if result["placement"] == 1:
    with col3:
        st.metric("Salary", f"{result['salary']:.2f} LPA")
    st.success("Peluang kerja bagus!")
else:
    with col3:
        st.metric("Salary", "N/A")
    st.warning("Perlu meningkatkan profil")