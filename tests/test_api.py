# tests/test_api.py
from fastapi.testclient import TestClient
import src.main as main

# Must match the "raw" input schema for /predict
PAYLOAD = {
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
    "thal": 3,
}

# For your API, these must match what training produced AFTER one-hot encoding
# Minimal safe schema: include numeric + all possible dummies that might appear.
FEATURE_COLUMNS = [
    # numeric
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    # categorical one-hot columns (a safe superset for dummy testing)
    "sex_0", "sex_1",
    "cp_0", "cp_1", "cp_2", "cp_3",
    "fbs_0", "fbs_1",
    "restecg_0", "restecg_1", "restecg_2",
    "exang_0", "exang_1",
    "slope_0", "slope_1", "slope_2",
    "ca_0", "ca_1", "ca_2", "ca_3",
    "thal_0", "thal_1", "thal_2", "thal_3",
    # sometimes included in training artifacts
    "disease_present",
]


def test_health():
    with TestClient(main.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}


def test_predict_structure():
    # Ensure test doesn't depend on a real model.pkl
    # Override model_artifacts directly for predict()
    main.model_artifacts = {
        "model": main.DummyModel(),
        "scaler": main.DummyScaler(),
        "feature_columns": FEATURE_COLUMNS,
    }

    with TestClient(main.app) as client:
        r = client.post("/predict", json=PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1]
    assert isinstance(body["confidence"], float)


def test_predict_invalid_data():
    with TestClient(main.app) as client:
        r = client.post("/predict", json={"age": 50})
    assert r.status_code == 422
