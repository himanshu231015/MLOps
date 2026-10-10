# Iris Classification — MLOps Day 2

## Overview

A beginner-friendly machine learning project that trains a Random Forest classifier to predict Iris flower species from four numerical measurements.

## Technologies

- Python
- scikit-learn
- Joblib
- pytest
- Git and GitHub

## Project Structure

```text
mlops-day1/
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── tests/
│   └── test_model.py
└── models/
    └── iris_model.joblib
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

The script evaluates the classifier and saves the trained artifact to `models/iris_model.joblib`.

## Make Predictions

```powershell
python predict.py
```

## Run Tests

```powershell
python -m pytest -v
```

## Reproducibility

The training script uses fixed random seeds and a stratified train/test split. Package versions are recorded in `requirements.txt`.

## Learning Objectives

- Train and evaluate a classification model
- Save and reload a model artifact
- Manage dependencies
- Test model behavior
- Track source code with Git and GitHub

## Limitations

This is a learning project using the Iris dataset. It is not a production-ready prediction service.# Iris Classification — MLOps Day 2

## Overview

A beginner-friendly machine learning project that trains a Random Forest classifier to predict Iris flower species from four numerical measurements.

## Technologies

- Python
- scikit-learn
- Joblib
- pytest
- Git and GitHub

## Project Structure

```text
mlops-day1/
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── tests/
│   └── test_model.py
└── models/
    └── iris_model.joblib
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

The script evaluates the classifier and saves the trained artifact to `models/iris_model.joblib`.

## Make Predictions

```powershell
python predict.py
```

## Run Tests

```powershell
python -m pytest -v
```

## Reproducibility

The training script uses fixed random seeds and a stratified train/test split. Package versions are recorded in `requirements.txt`.

## Learning Objectives

- Train and evaluate a classification model
- Save and reload a model artifact
- Manage dependencies
- Test model behavior
- Track source code with Git and GitHub

## Limitations

This is a learning project using the Iris dataset. It is not a production-ready prediction service.# Iris Classification — MLOps Day 2

## Overview

A beginner-friendly machine learning project that trains a Random Forest classifier to predict Iris flower species from four numerical measurements.

## Technologies

- Python
- scikit-learn
- Joblib
- pytest
- Git and GitHub

## Project Structure

```text
mlops-day1/
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── tests/
│   └── test_model.py
└── models/
    └── iris_model.joblib
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

The script evaluates the classifier and saves the trained artifact to `models/iris_model.joblib`.

## Make Predictions

```powershell
python predict.py
```

## Run Tests

```powershell
python -m pytest -v
```

## Reproducibility

The training script uses fixed random seeds and a stratified train/test split. Package versions are recorded in `requirements.txt`.

## Learning Objectives

- Train and evaluate a classification model
- Save and reload a model artifact
- Manage dependencies
- Test model behavior
- Track source code with Git and GitHub

## Limitations

This is a learning project using the Iris dataset. It is not a production-ready prediction service.
