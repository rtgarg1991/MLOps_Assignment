import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from gcputils import upload_directory_to_gcs, upload_file_to_gcs

# ================== PATHS ==================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"
APP_MODELS_DIR = PROJECT_ROOT / "app" / "models"

FEATURE_LIST_PATH = DATA_DIR / "feature_columns.txt"
METRICS_PATH = VISUALS_DIR / "metrics.csv"

APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================== GCP CONFIG ==================
GCP_BUCKET_NAME = DATA_DIR.name              # -> "data"
GCP_ARTIFACT_PREFIX = "artifacts/heart_disease"

# ================== CONFIG ==================
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "disease_present"

MODEL_FILES = {
    "Logistic Regression": APP_MODELS_DIR / "logistic_regression_model.pkl",
    "Random Forest": APP_MODELS_DIR / "random_forest_model.pkl",
}

# ================== SERIALIZATION ==================
def serialize_metadata(
    feature_names: List[str],
    metrics_df: pd.DataFrame
) -> Path:
    """
    Save preprocessing + evaluation metadata required for inference.
    """
    metadata = {
        "feature_names": feature_names,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": TARGET_COLUMN,
        "metrics": metrics_df.to_dict(orient="records"),
    }

    metadata_path = APP_MODELS_DIR / "preprocessing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved locally → {metadata_path}")
    return metadata_path

# ================== MAIN ==================
def main():
    print("Starting model packaging & serialization step...")

    # ---------- Load feature names ----------
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            "feature_columns.txt not found. Run feature_engineer.py first."
        )

    with open(FEATURE_LIST_PATH) as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(feature_names)} feature names")

    # ---------- Load metrics ----------
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "metrics.csv not found. Run train.py first."
        )

    metrics_df = pd.read_csv(METRICS_PATH)
    print("Loaded model evaluation metrics")

    # ---------- Verify trained models ----------
    loaded_models = {}
    for model_name, model_path in MODEL_FILES.items():
        if model_path.exists():
            loaded_models[model_name] = joblib.load(model_path)
            print(f"Loaded model → {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path} (skipping load)")

    if not loaded_models:
        print("⚠️ No trained models found to load (metadata will still be saved)")

    # ---------- Save metadata ----------
    metadata_path = serialize_metadata(feature_names, metrics_df)

    # ================== UPLOAD TO GCP ==================

    # Upload models directory (all .pkl + metadata json)
    upload_directory_to_gcs(
        local_dir=APP_MODELS_DIR,
        bucket_name=GCP_BUCKET_NAME,
        gcs_prefix=f"{GCP_ARTIFACT_PREFIX}/packaged/models"
    )

    # Upload metrics
    upload_file_to_gcs(
        local_file_path=METRICS_PATH,
        bucket_name=GCP_BUCKET_NAME,
        blob_path=f"{GCP_ARTIFACT_PREFIX}/packaged/metrics/metrics.csv"
    )

    # Upload feature list
    upload_file_to_gcs(
        local_file_path=FEATURE_LIST_PATH,
        bucket_name=GCP_BUCKET_NAME,
        blob_path=f"{GCP_ARTIFACT_PREFIX}/packaged/features/feature_columns.txt"
    )

    print("Model packaging & GCP upload completed successfully")

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    main()
