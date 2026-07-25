import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error


def evaluate(x_test_cls, y_test_cls, x_test_reg, y_test_reg, run_id):

    placement_model = mlflow.sklearn.load_model(f"runs:/{run_id}/placement_model")
    salary_model = mlflow.sklearn.load_model(f"runs:/{run_id}/salary_model")

    y_pred_cls = placement_model.predict(x_test_cls)

    acc = accuracy_score(y_test_cls, y_pred_cls)
    f1 = f1_score(y_test_cls, y_pred_cls)

    y_pred_reg = salary_model.predict(x_test_reg)

    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    print("\n=== Evaluation ===")
    print(f"Classification → Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print(f"Regression → R2: {r2:.3f}, MAE: {mae:.3f}")

    return acc, f1, r2, mae