
import json
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_saved_model_can_predict():
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    class_names = artifact["class_names"]

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(prediction) == 1
    assert int(prediction[0]) in range(len(class_names))


def test_metrics_are_valid():
    assert METRICS_PATH.exists()

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert metrics["random_state"] == 42
