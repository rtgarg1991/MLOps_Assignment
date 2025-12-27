import argparse
import pandas as pd
from datetime import datetime
from google.cloud import storage, bigquery

def apply_feature_engineering_logic(df):
    # Validation
    if 'target' not in df.columns:
        raise ValueError("Input data missing 'target' column. Pre-processing might be incorrect.")
    return df 

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