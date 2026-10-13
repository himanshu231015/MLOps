import sys
from pathlib import Path

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT

sys.path.insert(0, str(SOURCE_DIR))

from quality_gate import (
    apply_quality_gate,
    evaluate_classifier,
)


@pytest.fixture
def trained_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_evaluation_returns_valid_metrics(trained_model):
    model, X_test, y_test = trained_model

    metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0

    assert metrics["test_samples"] == len(y_test)
    assert len(metrics["confusion_matrix"]) == 3


def test_quality_gate_accepts_good_metrics():
    metrics = {
        "accuracy": 0.95,
        "macro_f1": 0.94,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "passed"
    assert result["failures"] == []


def test_quality_gate_rejects_low_accuracy():
    metrics = {
        "accuracy": 0.80,
        "macro_f1": 0.95,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "failed"
    assert any(
        "Accuracy" in failure
        for failure in result["failures"]
    )


def test_quality_gate_rejects_low_macro_f1():
    metrics = {
        "accuracy": 0.95,
        "macro_f1": 0.80,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "failed"
    assert any(
        "Macro F1" in failure
        for failure in result["failures"]
    )