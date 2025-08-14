from fastapi.testclient import TestClient

from src.api.main import PredictionOut, app  # Import PredictionOut for the assertion

client = TestClient(app)


# 1. We define a simple "mock" model class.
# It mimics the real model by having a .predict() method.
class MockModel:
    def predict(self, data):
        # It always returns a predictable result for our test.
        # Let's say it always predicts class '1' (Versicolor).
        import numpy as np

        return np.array([1])


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Iris model API is running."}


# 2. We add the 'monkeypatch' fixture as an argument to our test function.
def test_predict_endpoint(monkeypatch):
    # 3. We use monkeypatch to replace the real model in our 'main' module
    # with our fake MockModel for the duration of this single test.
    monkeypatch.setattr("src.api.main.model", MockModel())

    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)

    # 4. We update our assertions to check for the specific, predictable
    # output that we defined in our MockModel.
    assert response.status_code == 200

    # Check if the response body matches the Pydantic model for the expected output
    expected_data = PredictionOut(prediction=1, class_name="Versicolor")
    assert response.json() == expected_data.dict()
