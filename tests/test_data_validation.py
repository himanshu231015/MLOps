import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris


# Add the source directory to the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "DAY 4 — Experiment tracking"

sys.path.insert(0, str(SOURCE_DIR))

from validate_data import validate_dataset


@pytest.fixture
def iris_data():
    iris = load_iris()
    return iris.data.copy(), iris.target.copy()


def test_valid_dataset_passes(iris_data):
    X, y = iris_data

    result = validate_dataset(X, y)

    assert result["status"] == "passed"
    assert result["rows"] == 150
    assert result["features"] == 4


def test_missing_values_are_rejected(iris_data):
    X, y = iris_data

    X[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        validate_dataset(X, y)


def test_incorrect_feature_count_is_rejected(iris_data):
    X, y = iris_data

    X = X[:, :3]

    with pytest.raises(
        ValueError,
        match="Expected 4 features",
    ):
        validate_dataset(X, y)


def test_mismatched_row_counts_are_rejected(iris_data):
    X, y = iris_data

    y = y[:-1]

    with pytest.raises(
        ValueError,
        match="row counts do not match",
    ):
        validate_dataset(X, y)


def test_invalid_target_label_is_rejected(iris_data):
    X, y = iris_data

    y[0] = 5

    with pytest.raises(
        ValueError,
        match="Target labels must be 0, 1, or 2",
    ):
        validate_dataset(X, y)