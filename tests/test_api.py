from fastapi.testclient import TestClient

from src.train import train

train()  # make sure a model exists before the app loads it
from src.app import app  # noqa: E402

client = TestClient(app)

SAMPLE = {
    "age": 0.03, "sex": 0.05, "bmi": 0.06, "bp": 0.02, "s1": -0.04,
    "s2": -0.03, "s3": -0.04, "s4": 0.0, "s5": 0.02, "s6": -0.02,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_number():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_rejects_missing_field():
    bad = dict(SAMPLE)
    bad.pop("bmi")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
