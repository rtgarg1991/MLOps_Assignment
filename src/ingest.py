import argparse
import pandas as pd
from datetime import datetime
from google.cloud import bigquery

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    # 1. Load from local repo source
    local_path = "data/processed.cleveland.data"
    df = pd.read_csv(local_path, header=None)

    # 2. Save to GCS (Experimental Raw folder)
    gcs_uri = f"gs://{args.bucket}/data/raw/pr-{args.pr_number}/raw.csv"
    df.to_csv(gcs_uri, index=False, header=False)
    print(f"Ingested raw data to: {gcs_uri}")

    # 3. Log metadata to BigQuery
    client_bq = bigquery.Client()
    table_id = f"{client_bq.project}.ml_metadata.data_versions"
    tracking_row = {
        "data_type": "raw", "pr_number": int(args.pr_number), "gcs_path": gcs_uri,
        "status": "experiment", "created_at": datetime.now().isoformat()
    }
    client_bq.insert_rows_json(table_id, [tracking_row])

if __name__ == "__main__":
    main()