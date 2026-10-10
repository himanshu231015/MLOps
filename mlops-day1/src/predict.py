
import joblib

# 1. Load the saved artifact
artifact = joblib.load("models/iris_model.joblib")

model = artifact["model"]
class_names = artifact["class_names"]

# 2. Measurements of a new flower
# Order: sepal length, sepal width, petal length, petal width
flower = [[5.1, 3.5, 1.4, 0.2]]

# 3. Predict the species
prediction = model.predict(flower)[0]

# 4. Convert the numeric label into a species name
species = class_names[prediction]

print("Predicted flower species:", species)
