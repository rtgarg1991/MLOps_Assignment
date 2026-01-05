import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from google.cloud import storage, bigquery

# Define columns since raw data has no header
COLUMNS = [
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
    "target",
]

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"

# ------------------ CONFIG ------------------
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "num"


def apply_preprocessing_logic(df):
    # Assign headers to the raw dataframe
    df.columns = COLUMNS

    # Basic cleaning
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Ensure numeric types
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col])

    return df


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    client_gcs = storage.Client()
    bucket = client_gcs.bucket(args.bucket)

    # LOGIC GATE: Use PR Raw if available, else Production
    pr_raw = f"data/raw/pr-{args.pr_number}/raw.csv"
    input_path = (
        pr_raw
        if bucket.blob(pr_raw).exists()
        else "data/raw/production/latest.csv"
    )

    print(f"Pre-processing input: gs://{args.bucket}/{input_path}")

    # Read raw data
    df = pd.read_csv(f"gs://{args.bucket}/{input_path}")

    # Clean Data
    df_processed = clean_data(df)

    output_uri = (
        f"gs://{args.bucket}/data/processed/pr-{args.pr_number}/processed.csv"
    )

    # SAVE WITH HEADER so feature_engineering.py can read it easily
    df_processed.to_csv(output_uri, index=False)
    print(f"Saved processed data to {output_uri}")

    # Log to BigQuery
    client_bq = bigquery.Client()
    bigquery.Client().insert_rows_json(
        f"{client_bq.project}.ml_metadata.data_versions",
        [
            {
                "data_type": "processed",
                "pr_number": int(args.pr_number),
                "gcs_path": output_uri,
                "status": "experiment",
                "created_at": datetime.now().isoformat(),
            }
        ],
    )


if __name__ == "__main__":
    main()
