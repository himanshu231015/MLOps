import csv
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Resolve paths independently of the directory from which
# this script is executed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent

MODEL_DIR = SOURCE_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "experiments.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load the dataset.
dataset = load_iris()
X = dataset.data
y = dataset.target


# 2. Create one fixed train/test split for both experiments.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Define the experiment results we want to record.
fieldnames = [
    "timestamp_utc",
    "run_id",
    "run_name",
    "dataset",
    "test_size",
    "random_state",
    "n_estimators",
    "accuracy",
    "model_path",
]


# 4. Run two controlled experiments.
for n_estimators in (50, 150):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
    )

    # Train the model.
    model.fit(X_train, y_train)

    # Evaluate the model on unseen test observations.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Give this run a unique ID.
    now = datetime.now(timezone.utc)

    timestamp_utc = now.isoformat()
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"iris_rf_{n_estimators}"

    # Save a separate model artifact for this run.
    model_path = MODEL_DIR / f"{run_name}_{run_id}.joblib"

    joblib.dump(
        {
            "model": model,
            "class_names": list(dataset.target_names),
        },
        model_path,
    )

    # Record the experiment metadata and results.
    row = {
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "run_name": run_name,
        "dataset": "sklearn.datasets.load_iris",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": n_estimators,
        "accuracy": round(float(accuracy), 4),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }

    # Append the result to the CSV file.
    with METRICS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(row)

    print(f"\nExperiment: {run_name}")
    print(f"Run ID: {run_id}")
    print(f"Number of trees: {n_estimators}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics recorded in: {METRICS_FILE}")

print("\nAll experiments completed.")