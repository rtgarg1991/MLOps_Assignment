import os
import pickle
import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("heart-api")

MODEL_PATH = os.path.join(os.getcwd(), "models/model.pkl")

model_artifacts = None

PREDICTION_COUNTER = Counter(
    "heart_prediction_total",
    "Total number of predictions made",
    ["pred_class"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "heart_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

INFERENCE_LATENCY = Histogram(
    "heart_model_processing_seconds",
    "Time spent specifically on model inference",
)

AGE_DISTRIBUTION = Histogram(
    "heart_input_age_distribution",
    "Distribution of patient ages in requests",
    buckets=[20, 30, 40, 50, 60, 70, 80],
)

RESTING_BP_DISTRIBUTION = Histogram(
    "heart_input_resting_bp_distribution",
    "Distribution of resting blood pressure",
    buckets=[80, 100, 120, 140, 160, 180, 200],
)

CHOLESTEROL_DISTRIBUTION = Histogram(
    "heart_input_cholesterol_distribution",
    "Distribution of serum cholesterol levels",
    buckets=[100, 150, 200, 250, 300, 350, 400],
)

MAX_HEART_RATE_DISTRIBUTION = Histogram(
    "heart_input_max_heart_rate_distribution",
    "Distribution of maximum heart rate achieved",
    buckets=[60, 80, 100, 120, 140, 160, 180, 200],
)


class DummyScaler:
    def transform(self, X):
        return X


class DummyModel:
    def predict(self, X):
        import numpy as np

        return np.array([1])

    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.05, 0.95]])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifacts

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_artifacts = pickle.load(f)

        logger.info(f"Successfully loaded real model from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Using DUMMY model.")
        model_artifacts = {
            "model": DummyModel(),
            "scaler": DummyScaler(),
            "feature_columns": [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ],
        }

    yield
    model_artifacts = None


app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Duration: {process_time:.2f}ms"
    )
    return response


class PredictionInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PredictionInput):
    if model_artifacts is None:
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded")

    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    feature_columns = [
        c
        for c in model_artifacts["feature_columns"]
        if c != "disease_present"
    ]

    input_dict = data.model_dump()
    logger.info(f"Received prediction request: {input_dict}")

    AGE_DISTRIBUTION.observe(input_dict["age"])
    MAX_HEART_RATE_DISTRIBUTION.observe(input_dict["thalach"])
    RESTING_BP_DISTRIBUTION.observe(input_dict["trestbps"])
    CHOLESTEROL_DISTRIBUTION.observe(input_dict["chol"])

    input_df = pd.DataFrame([input_dict])
    categorical_cols = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    try:
        with INFERENCE_LATENCY.time():
            # Applying SAME one-hot encoding as training
            input_encoded = pd.get_dummies(
                input_df, columns=categorical_cols, drop_first=False
            )

            # Align with training feature schema
            input_encoded = input_encoded.reindex(
                columns=feature_columns, fill_value=0
            )

            input_encoded = input_encoded.astype(int)

            # Applying scaling only if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(input_encoded)
            else:
                X_scaled = input_encoded
            prediction = int(model.predict(X_scaled)[0])

            if hasattr(model, "predict_proba"):
                confidence = float(
                    model.predict_proba(X_scaled)[0][prediction]
                )
            else:
                confidence = 1.0

        PREDICTION_COUNTER.labels(pred_class=str(prediction)).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)

        result = {"prediction": prediction, "confidence": confidence}

        logger.info(f"Prediction success: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
