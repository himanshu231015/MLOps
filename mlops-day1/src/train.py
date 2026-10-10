from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


# 1. Load Dataset.

iris = load_iris()
X = iris.data
Y = iris.target


# 2. Split the data into training and testing.
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 3. Create the ML Model.

Model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Train the model.
Model.fit(X_train,Y_train)

# 5. Evaluate the model on unseen dataset.
y_pred = Model.predict(X_test)


accuracy = accuracy_score(Y_test,y_pred)


print(f"Model accuracy: {accuracy:.2%}")
print("\nClassification report:")

print(
    classification_report(
        Y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# 6. Save the trained model and class names

Path("models").mkdir(exist_ok=True)


joblib.dump(
    {
        "model":Model,
        "class_names": iris.target_names.tolist()
    },
    "models/iris_model.joblib"
)
print("\nModel saved to models/iris_model.joblib")
