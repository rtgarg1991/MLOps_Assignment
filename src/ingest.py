import argparse
import pandas as pd
from datetime import datetime
from google.cloud import bigquery
from pathlib import Path


# ================== PROJECT ROOT ==================
PROJECT_ROOT = Path.cwd()

# ================== CORE DIRECTORIES ==================
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"


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
    df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in df.columns
    ]

    # 🔹 Saving dataset
    file_path = DATA_DIR / "heart_disease_raw.csv"
    df.to_csv(file_path, index=False)

    print(f"Dataset saved to: {file_path}")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("ingest done")

    return df


def create_directories() -> None:
    """
    Create all required project directories.
    """
    core_dirs = [DATA_DIR, VISUALS_DIR]
    for directory in core_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    # 1. Create the directories
    create_directories()

    # 2. Ingest data
    ingest_data()

    # 2. Load from local repo source
    local_path = DATA_DIR / "heart_disease_raw.csv"
    df = pd.read_csv(local_path)

    # 2. Save to GCS (Experimental Raw folder)
    gcs_uri = f"gs://{args.bucket}/data/raw/pr-{args.pr_number}/raw.csv"
    df.to_csv(gcs_uri, index=False)
    print(f"Ingested raw data to: {gcs_uri}")

    # 3. Log metadata to BigQuery
    client_bq = bigquery.Client()
    table_id = f"{client_bq.project}.ml_metadata.data_versions"
    tracking_row = {
        "data_type": "raw",
        "pr_number": int(args.pr_number),
        "gcs_path": gcs_uri,
        "status": "experiment",
        "created_at": datetime.now().isoformat(),
    }

    client_bq.insert_rows_json(table_id, [tracking_row])


if __name__ == "__main__":
    main()
