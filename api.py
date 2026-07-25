from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Student Placement API")

placement_model = joblib.load("artifacts/placement_model.pkl")
salary_model = joblib.load("artifacts/salary_model.pkl")

class Student(BaseModel):
    gender: str
    ssc_percentage: int
    hsc_percentage: int
    degree_percentage: int
    cgpa: float
    entrance_exam_score: int
    technical_skill_score: int
    soft_skill_score: int
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: int
    backlogs: int
    extracurricular_activities: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: Student):

    df = pd.DataFrame([data.dict()])

    df["total_skills"] = df["technical_skill_score"] + df["soft_skill_score"]
    df["experience_score"] = df["internship_count"] + df["work_experience_months"] / 12

    placement = int(placement_model.predict(df)[0])
    proba = float(placement_model.predict_proba(df)[0][1])

    result = {
        "placement": placement,
        "probability": round(proba, 3)
    }

    if placement == 1:
        salary = float(salary_model.predict(df)[0])
        result["salary"] = round(salary, 2)
    else:
        result["salary"] = None

    return result