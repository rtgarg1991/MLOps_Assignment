# tests/test_api.py
from fastapi.testclient import TestClient
from src.main import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}

def test_predict_structure():
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "confidence" in body
        assert body["prediction"] in [0, 1]
        assert isinstance(body["confidence"], float)

def test_predict_invalid_data():
    with TestClient(app) as client:
        r = client.post("/predict", json={"age": 50})
        assert r.status_code == 422
