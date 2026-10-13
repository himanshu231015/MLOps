import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris


def validate_dataset(X, y, expected_features=4):
    """
    Validate feature data and target labels.

    Raises:
        ValueError: If the dataset violates validation rules.

    Returns:
        dict: Summary of the validated dataset.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    errors = []

    # 1. Validate feature dimensions.
    if X.ndim != 2:
        errors.append(
            "Features must be a two-dimensional array."
        )

    elif X.shape[1] != expected_features:
        errors.append(
            f"Expected {expected_features} features, "
            f"but received {X.shape[1]}."
        )

    # 2. Reject an empty dataset.
    if X.ndim == 2 and X.shape[0] == 0:
        errors.append("Dataset contains no observations.")

    # 3. Validate feature data types and numerical values.
    if not np.issubdtype(X.dtype, np.number):
        errors.append("Features must contain numeric values.")

    elif not np.isfinite(X).all():
        errors.append(
            "Features contain missing or non-finite values."               
        )

    # 4. Validate target dimensions.
    if y.ndim != 1:
        errors.append(
            "Target labels must be a one-dimensional array."
        )

    # 5. Validate feature and target consistency.
    if X.ndim == 2 and y.ndim == 1:
        if len(X) != len(y):
            errors.append(
                "Feature and target row counts do not match."
            )

    # 6. Validate target labels.
    allowed_labels = np.array([0, 1, 2])

    if y.ndim == 1:
        if not np.isin(y, allowed_labels).all():
            errors.append(
                "Target labels must be 0, 1, or 2."
            )

    # Fail fast if any rule is violated.
    if errors:
        raise ValueError(
            "Dataset validation failed:\n- "
            + "\n- ".join(errors)
        )

    # Return a summary if validation succeeds.
    return {
        "status": "passed",
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "classes": sorted(
            np.unique(y).astype(int).tolist()
        ),
    }


def main():
    # Load the Iris dataset.
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Validate the dataset before training.
    summary = validate_dataset(X, y)

    # Print the validation results.
    print("Data validation successful.")
    print(json.dumps(summary, indent=4))

    # Save a validation report.
    project_root = Path(__file__).resolve().parents[0]
    reports_dir = project_root / "reports"

    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "data_validation.json"

    report_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()