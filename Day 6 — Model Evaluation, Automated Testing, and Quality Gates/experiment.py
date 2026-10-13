
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from quality_gate import apply_quality_gate, evaluate_classifier


# --------------------------------------------------
# 1. Resolve project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent

MODEL_DIR = SOURCE_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "experiments_day6.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------
dataset = load_iris()
X = dataset.data
y = dataset.target


# --------------------------------------------------
# 3. Create a fixed train/test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 4. Define experiment tracking columns
# --------------------------------------------------
fieldnames = [
    "timestamp_utc",
    "run_id",
    "run_name",
    "dataset",
    "test_size",
    "random_state",
    "n_estimators",
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "quality_gate",
    "gate_failures",
    "confusion_matrix",
    "model_path",
]

accepted_runs = 0


# --------------------------------------------------
# 5. Run controlled experiments
# --------------------------------------------------
for n_estimators in (50, 150):

    # Train a separate candidate for each experiment.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate the trained candidate.
    evaluation = evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    # Apply the quality gate.
    gate = apply_quality_gate(evaluation)

    now = datetime.now(timezone.utc)
    timestamp_utc = now.isoformat()
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"iris_rf_{n_estimators}"

    model_path = MODEL_DIR / f"{run_name}_{run_id}.joblib"

    # Save artifacts only for accepted candidates.
    if gate["status"] == "passed":
        joblib.dump(
            {
                "model": model,
                "class_names": list(dataset.target_names),
            },
            model_path,
        )

        accepted_runs += 1
        saved_model_path = str(
            model_path.relative_to(PROJECT_ROOT)
        )
    else:
        saved_model_path = ""

    # Record every experiment, even rejected ones.
    row = {
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "run_name": run_name,
        "dataset": "sklearn.datasets.load_iris",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": n_estimators,
        "accuracy": round(evaluation["accuracy"], 4),
        "macro_f1": round(evaluation["macro_f1"], 4),
        "macro_precision": round(
            evaluation["macro_precision"], 4
        ),
        "macro_recall": round(
            evaluation["macro_recall"], 4
        ),
        "quality_gate": gate["status"],
        "gate_failures": " | ".join(gate["failures"]),
        "confusion_matrix": json.dumps(
            evaluation["confusion_matrix"]
        ),
        "model_path": saved_model_path,
    }

    # Write the header only when the CSV is new or empty.
    write_header = (
        not METRICS_FILE.exists()
        or METRICS_FILE.stat().st_size == 0
    )

    with METRICS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print(f"\nExperiment: {run_name}")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro F1: {evaluation['macro_f1']:.4f}")
    print(f"Quality gate: {gate['status']}")

    if gate["failures"]:
        for failure in gate["failures"]:
            print(f"Rejection reason: {failure}")

    if saved_model_path:
        print(f"Accepted model saved to: {saved_model_path}")
    else:
        print("Candidate rejected; model artifact not saved.")


# --------------------------------------------------
# 6. Stop the pipeline if every candidate is rejected
# --------------------------------------------------
if accepted_runs == 0:
    raise SystemExit(
        "No model passed the quality gate. "
        "Inspect the experiment results before continuing."
    )

print(f"\nAccepted model candidates: {accepted_runs}")
print(f"Experiment report: {METRICS_FILE}")
