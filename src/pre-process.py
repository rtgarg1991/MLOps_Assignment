import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gcputils import upload_file_to_gcs, upload_directory_to_gcs

# ------------------ PATHS ------------------
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"

RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"

# ================== GCP CONFIG ==================
GCP_BUCKET_NAME = DATA_DIR.name      
GCP_DATA_PREFIX = "datasets/heart_disease"
GCP_ARTIFACT_PREFIX = "artifacts/heart_disease"

# ------------------ CONFIG ------------------
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "num"

# ------------------ EDA ------------------
def run_eda(df: pd.DataFrame):
    print("Running EDA (before cleaning)...")

    # Missing values
    missing = df.isna().sum()
    missing[missing > 0].plot(kind="bar", title="Missing Values")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "missing_values/missing_values.png")
    plt.close()

    # Numeric distributions
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(VISUALS_DIR / "distributions" / f"{col}.png")
            plt.close()

    # Correlation heatmap
    corr = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "correlations/correlation_heatmap.png")
    plt.close()

# ------------------ CLEANING ------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")

    df_clean = df.copy()

    df_clean.replace("?", np.nan, inplace=True)

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    for col in ["ca", "thal"]:
        if col in df_clean.columns:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)

    if TARGET_COLUMN in df_clean.columns:
        df_clean["disease_present"] = df_clean[TARGET_COLUMN].apply(
            lambda x: 1 if x > 0 else 0
        )

    return df_clean
    
    

# ------------------ MAIN ------------------
def main():

    print(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    run_eda(df)

    df_clean = clean_data(df)

    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_DATA_PATH}")
    # Upload cleaned dataset to GCS
    upload_file_to_gcs(
        local_file_path=CLEAN_DATA_PATH,
        bucket_name=GCP_BUCKET_NAME,
        blob_path=f"{GCP_DATA_PREFIX}/heart_disease_clean.csv"
    )
    
     # ---------- UPLOAD EDA ARTIFACTS ----------
    upload_directory_to_gcs(
        local_dir=VISUALS_DIR,
        bucket_name=GCP_BUCKET_NAME,
        gcs_prefix=f"{GCP_ARTIFACT_PREFIX}/visuals"
    )
     

if __name__ == "__main__":
    main()
