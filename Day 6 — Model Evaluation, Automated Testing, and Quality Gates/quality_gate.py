from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a fitted classifier on held-out test data.
    """

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "test_samples": int(len(y_test)),
    }

    return metrics


def apply_quality_gate(
    metrics,
    min_accuracy=0.90,
    min_macro_f1=0.90,
):
    """
    Determine whether a model meets the required thresholds.
    """

    failures = []

    if metrics["accuracy"] < min_accuracy:
        failures.append(
            f"Accuracy {metrics['accuracy']:.4f} "
            f"is below {min_accuracy:.4f}."
        )

    if metrics["macro_f1"] < min_macro_f1:
        failures.append(
            f"Macro F1 {metrics['macro_f1']:.4f} "
            f"is below {min_macro_f1:.4f}."
        )

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_macro_f1": min_macro_f1,
        },
    }