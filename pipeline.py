from data_ingestion import ingest_data
from train_pipeline import train_models, feature_engineering
from evaluation import evaluate

import pandas as pd
from sklearn.model_selection import train_test_split

def run_pipeline():

    print("Step 1: Data Ingestion")
    ingest_data()
    df = pd.read_csv("ingested/B.csv")

    df = feature_engineering(df)
    df = df.drop(columns=["student_id"])

    x = df.drop(columns=["placement_status", "salary_package_lpa"])
    y_cls = df["placement_status"]
    x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
        x, y_cls,
        test_size=0.2,
        stratify=y_cls,
        random_state=42
    )

    df_reg = df[df["placement_status"] == 1]
    x_reg = df_reg.drop(columns=["placement_status", "salary_package_lpa"])
    y_reg = df_reg["salary_package_lpa"]
    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x_reg, y_reg,
        test_size=0.2,
        random_state=42
    )

    print("Step 2: Training")
    run_id = train_models(x_train_cls, y_train_cls, x_train_reg, y_train_reg)

    print("Step 3: Evaluation")
    evaluate(x_test_cls, y_test_cls, x_test_reg, y_test_reg, run_id)


if __name__ == "__main__":
    run_pipeline()