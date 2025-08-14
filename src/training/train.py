import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def train_and_register_model():
    """
    This function trains multiple models, logs them with MLflow,
    identifies the best one, and promotes it for production use
    by setting a 'production' alias.
    """
    # --- MLflow Setup ---
    # Set the tracking URI to a local directory. This is the portable way.
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Iris Classification")

    print("Loading data...")
    df = pd.read_csv("data/raw/iris.csv")

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define an input example to save with the model for schema inference.
    # This is a best practice for model logging.
    input_example = X_train.head(1)

    # --- Model 1: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic Regression"):
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            sk_model=lr, artifact_path="model", input_example=input_example
        )

    # --- Model 2: Random Forest ---
    with mlflow.start_run(run_name="Random Forest"):
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            sk_model=rf, artifact_path="model", input_example=input_example
        )

    # --- Model Registration and Promotion ---
    print("Registering the best model...")
    client = mlflow.tracking.MlflowClient()

    # Find the best run based on accuracy
    runs = client.search_runs(
        experiment_ids=mlflow.get_experiment_by_name(
            "Iris Classification"
        ).experiment_id,
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        raise Exception("No runs found for this experiment.")

    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "IrisClassifier"

    # Register the model from the best run
    model_version = mlflow.register_model(model_uri=best_model_uri, name=model_name)
    print(f"Model '{model_name}' version {model_version.version} registered.")

    # Use the modern alias method to promote the model
    print(f"Setting alias 'production' to model version {model_version.version}")
    client.set_registered_model_alias(
        name=model_name, alias="production", version=model_version.version
    )
    print("Alias 'production' set successfully.")


if __name__ == "__main__":
    train_and_register_model()
