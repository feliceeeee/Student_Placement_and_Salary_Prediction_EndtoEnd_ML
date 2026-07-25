import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

placement_model = joblib.load("artifacts/placement_model.pkl")
salary_model = joblib.load("artifacts/salary_model.pkl")

# config
st.set_page_config(page_title="Student Placement", layout="wide")

# sidebar
st.sidebar.title("Info")
st.sidebar.write("Prediksi peluang kerja & estimasi gaji mahasiswa.")
if st.sidebar.button("Reset"):
    st.rerun()

# title
st.title("Student Placement & Salary Prediction")
st.caption("Isi data mahasiswa untuk melihat peluang kerja dan estimasi gaji.")

# form input
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

    submit = st.form_submit_button("Predict")

# prediction
if not submit:
    st.info("Silakan isi form di atas lalu klik **Predict**")
    st.stop()

# create input data
input_data = pd.DataFrame([{
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
}])

input_data["total_skills"] = tech + soft
input_data["experience_score"] = internship + experience / 12

st.divider()

# profile comparison 
st.subheader("Profile Comparison")
features = ["cgpa", "technical_skill_score", "soft_skill_score", "backlogs"]
labels = ["CGPA", "Tech Skill", "Soft Skill", "Backlogs"]

user_vals = [input_data[f].values[0] for f in features]

# bar chart
fig, ax = plt.subplots(figsize=(6,3))
ax.bar(labels, user_vals)
ax.set_title("Your Key Features")
plt.tight_layout()
st.pyplot(fig)

# prediction result
st.subheader("Prediction Result")
placement = placement_model.predict(input_data)[0]
proba = placement_model.predict_proba(input_data)[0][1]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Placement", "Placed" if placement == 1 else "Not Placed")
with col2:
    st.metric("Confidence", f"{proba:.2%}")

if placement == 1:
    salary = salary_model.predict(input_data)[0]
    with col3:
        st.metric("Salary", f"{salary:.2f} LPA")
    st.success("Peluang kerja bagus!")
else:
    with col3:
        st.metric("Salary", "N/A")
    st.warning("Perlu meningkatkan profil")