from pathlib import Path
import pandas as pd
from gcputils import upload_file_to_gcs

# ================== PROJECT ROOT ==================
PROJECT_ROOT = Path.cwd()

# ================== CORE DIRECTORIES ==================
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORT_DIR = PROJECT_ROOT / "report"
MODELS_DIR = PROJECT_ROOT / "models"
APP_MODELS_DIR = PROJECT_ROOT / "app" / "models"

# ================== DATA FILE PATHS ==================
RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"

# ================== VISUAL SUBDIRECTORIES ==================
VISUAL_DISTRIBUTIONS_DIR = VISUALS_DIR / "distributions"
VISUAL_CORRELATIONS_DIR = VISUALS_DIR / "correlations"
VISUAL_MISSING_DIR = VISUALS_DIR / "missing_values"

# ================== GCP CONFIG ==================
GCP_BUCKET_NAME = DATA_DIR.name 
GCP_DATA_PREFIX = "datasets/heart_disease"

# ================== DIRECTORY CREATION ==================
def create_directories() -> None:
    """
    Create all required project directories.
    """
    core_dirs = [
        DATA_DIR,
        VISUALS_DIR,
        REPORT_DIR,
        MODELS_DIR,
        APP_MODELS_DIR,
        VISUAL_DISTRIBUTIONS_DIR,
        VISUAL_CORRELATIONS_DIR,
        VISUAL_MISSING_DIR,
    ]

    for directory in core_dirs:
        directory.mkdir(parents=True, exist_ok=True)
  

# Dataset configuration
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "num"
    
    

def ingest_data() -> pd.DataFrame:
    """
    Fetch UCI Heart Disease dataset
    """
    print("Ingesting data...")
    print("Fetching UCI Heart Disease dataset...")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features
    targets = dataset.data.targets

    # Merging features and target
    df = pd.concat([features, targets], axis=1)

    # Standardizing column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # 🔹 Saving dataset 
    file_path = DATA_DIR / "heart_disease_raw.csv"
    df.to_csv(file_path, index=False)

    print(f"Dataset saved to: {file_path}")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    
    # ---------- UPLOAD TO GCP ----------
    upload_file_to_gcs(
        local_file_path=RAW_DATA_PATH,
        bucket_name=GCP_BUCKET_NAME,
        blob_path=f"{GCP_DATA_PREFIX}/heart_disease_raw.csv"
    )

    return df

if __name__ == "__main__":
    create_directories()
    ingest_data()
    