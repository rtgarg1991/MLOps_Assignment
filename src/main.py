import os
import pickle
import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

# --- 1. SETUP LOGGING ---
# Configure logging to print to stdout (Google Cloud captures this automatically)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("heart-api")

MODEL_PATH = os.path.join(os.getcwd(), "models/model.pkl")

# Global variables
model_artifacts = None

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
            "scaler": DummyScaler()
        }

    yield
    model_artifacts = None

app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

# --- 2. ADD PROMETHEUS METRICS ---
# This automatically creates a /metrics endpoint for Prometheus to scrape
Instrumentator().instrument(app).expose(app)

# --- 3. ADD LOGGING MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate duration
    process_time = (time.time() - start_time) * 1000  # in milliseconds
    
    # Log details: Method, Path, Status, Duration
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
    
    input_dict = data.model_dump()
    logger.info(f"Received prediction request: {input_dict}")

    input_df = pd.DataFrame([input_dict])
    
    feature_cols = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ]
    input_df = input_df[feature_cols]
    
    try:
        X_scaled = scaler.transform(input_df)
        prediction = int(model.predict(X_scaled)[0])
        
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(X_scaled)[0][prediction])
        else:
            confidence = 1.0
            
        result = {
            "prediction": prediction,
            "confidence": confidence
        }
        
        logger.info(f"Prediction success: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)