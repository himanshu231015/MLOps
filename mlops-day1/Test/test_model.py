from pathlib import Path
import joblib
Model_path = Path("model/iris_model.joblib")

def test_saved_model_exists():
    assert Model_path.exists()

def test_saved_model_predicts_valid_class():
    artifact = joblib.load(Model_path)

    model = artifact["model"]
    class_names = artifact["class_names"]


    flower = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(flower)
    assert len(prediction) == 1
    assert int(prediction[0]) in range(len(class_names))