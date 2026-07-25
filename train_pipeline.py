import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def feature_engineering(df):
    df = df.copy()
    df["total_skills"] = df["technical_skill_score"] + df["soft_skill_score"]
    df["experience_score"] = df["internship_count"] + df["work_experience_months"] / 12
    return df

def build_preprocessor(X):

    num_feat = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feat = X.select_dtypes(include=['object']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer([
        ('num', numeric_pipeline, num_feat),
        ('cat', categorical_pipeline, cat_feat)
    ])

    return preprocess


def train_models(x_train_cls, y_train_cls, x_train_reg, y_train_reg):

    mlflow.set_experiment("Student Placement")

    preprocess_cls = build_preprocessor(x_train_cls)
    preprocess_reg = build_preprocessor(x_train_reg)

    placement_pipeline = Pipeline([
        ('preprocessing', preprocess_cls),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])

    salary_pipeline = Pipeline([
        ('preprocessing', preprocess_reg),
        ('model', LinearRegression())
    ])

    with mlflow.start_run() as run:

        placement_pipeline.fit(x_train_cls, y_train_cls)
        salary_pipeline.fit(x_train_reg, y_train_reg)

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(placement_pipeline, "artifacts/placement_model.pkl")
        joblib.dump(salary_pipeline, "artifacts/salary_model.pkl")

        mlflow.sklearn.log_model(placement_pipeline, "placement_model")
        mlflow.sklearn.log_model(salary_pipeline, "salary_model")

        mlflow.log_param("classification_model", "RandomForest")
        mlflow.log_param("regression_model", "LinearRegression")

    return run.info.run_id