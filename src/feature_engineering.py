import pandas as pd
from pathlib import Path
from typing import List, Tuple
import argparse
from datetime import datetime
from google.cloud import storage, bigquery

# ================== PATHS ==================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"
FEATURE_DATA_PATH = DATA_DIR / "heart_disease_features.csv"
FEATURE_LIST_PATH = DATA_DIR / "feature_columns.txt"

GCP_BUCKET_NAME = DATA_DIR.name        
GCP_DATA_PREFIX = "datasets/heart_disease"

# ================== CONFIG ==================
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "disease_present"

def apply_feature_engineering_logic(df):
    return one_hot_encode_features(df) 

# ================== FEATURE ENGINEERING ==================
def one_hot_encode_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical features and retain target column.

    Args:
        df: Cleaned dataframe (includes target column)
        categorical_cols: List of categorical feature names

    Returns:
        Tuple of:
        - Feature dataframe INCLUDING target column
        - List of feature column names (excluding target)
    """
    print("One-hot encoding categorical features...")

    # Separate columns
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in categorical_cols if c in df.columns]

    df_numeric = df[numeric_cols].copy()

    if cat_cols:
        df_cat = df[cat_cols].astype(str)
        df_cat_encoded = pd.get_dummies(df_cat, prefix=cat_cols, drop_first=False)
        df_features = pd.concat([df_numeric, df_cat_encoded], axis=1)
    else:
        df_features = df_numeric

    feature_columns = list(df_features.columns)

    # Add target column
    df_final = pd.concat(
        [df_features, df[[TARGET_COLUMN]]],
        axis=1
    )

    print(f"  Total features: {len(feature_columns)}")
    print(f"  Final dataset shape: {df_final.shape}")
    
    # Saving feature dataset
    df_features.to_csv(FEATURE_DATA_PATH, index=False)
    print(f"Feature dataset saved to {FEATURE_DATA_PATH}")

    # Saving feature column names
    with open(FEATURE_LIST_PATH, "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    print(f"Feature column list saved to {FEATURE_LIST_PATH}")

    return df_final
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    client_gcs = storage.Client()
    bucket = client_gcs.bucket(args.bucket)

    # LOGIC GATE: Use PR Processed if available, else Production
    pr_proc = f"data/processed/pr-{args.pr_number}/processed.csv"
    input_path = pr_proc if bucket.blob(pr_proc).exists() else "data/processed/production/latest.csv"
    
    print(f"Feature Engineering input: gs://{args.bucket}/{input_path}")
    
    # Read WITH header (since pre-processing.py now adds it)
    df = pd.read_csv(f"gs://{args.bucket}/{input_path}")
    df_features = apply_feature_engineering_logic(df)

    output_uri = f"gs://{args.bucket}/data/features/pr-{args.pr_number}/features.csv"
    df_features.to_csv(output_uri, index=False)
    print(f"Saved features to {output_uri}")

    # Log to BigQuery
    bigquery.Client().insert_rows_json(f"{bigquery.Client().project}.ml_metadata.data_versions", [{
        "data_type": "features", 
        "pr_number": int(args.pr_number),
        "gcs_path": output_uri, 
        "status": "experiment", 
        "created_at": datetime.now().isoformat()
    }])

if __name__ == "__main__":
    main()    
