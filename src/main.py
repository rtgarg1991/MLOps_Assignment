import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

MODEL_PATH = os.path.join(os.getcwd(), "models/model.pkl")

# Global variables for model and scaler
model_artifacts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifacts
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model_artifacts = pickle.load(f)
    yield
    model_artifacts = None

app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

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
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    
    # Convert input to DataFrame for scaler
    # Using model_dump() for Pydantic v2
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Ensure column order matches training
    feature_cols = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ]
    input_df = input_df[feature_cols]
    
    try:
        # Scale features
        X_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = int(model.predict(X_scaled)[0])
        
        # Get confidence if available
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(X_scaled)[0][prediction])
        else:
            confidence = 1.0 # Fallback for mock models without proba
            
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
