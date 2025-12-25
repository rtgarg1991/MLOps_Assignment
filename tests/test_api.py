import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_structure():
    # Example data based on the dataset
    payload = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0
    }
    
    # Use the test client; it triggers startup events
    with TestClient(app) as local_client:
        response = local_client.post("/predict", json=payload)
        
        # Check status code
        assert response.status_code == 200
        
        # Check structure
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0

def test_predict_invalid_data():
    payload = {
        "age": "invalid", # Should trigger validation error
        "sex": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
