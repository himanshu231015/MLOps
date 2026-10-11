
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

# Reproducibility settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Split before training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Train the model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 4. Evaluate on held-out test data
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_estimators": N_ESTIMATORS,
        "model_type": "RandomForestClassifier",
        "dataset": "Iris",
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=iris.target_names.tolist(),
            output_dict=True,
        ),
    }

    # 5. Save model artifact
    joblib.dump(
        {
            "model": model,
            "class_names": iris.target_names.tolist(),
        },
        MODEL_PATH,
    )

    # 6. Save metrics and experiment configuration
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
