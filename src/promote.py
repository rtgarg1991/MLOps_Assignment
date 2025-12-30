import argparse
from datetime import datetime
from google.cloud import storage, bigquery


def promote_blob(bucket, source_path, dest_path):
    source_blob = bucket.blob(source_path)
    if source_blob.exists():
        bucket.copy_blob(source_blob, bucket, dest_path)
        print(f"Promoted {source_path} to {dest_path}")
    else:
        print(f"WARNING: Source {source_path} not found. Skipping promotion.")


def log_promotion_to_bq(project_id, pr_number, data_type, gcs_path):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.ml_metadata.data_versions"

    # We INSERT a new row for production status instead of Updating
    # This avoids the "Streaming Buffer" error
    row = {
        "data_type": data_type,
        "pr_number": pr_number,
        "raw_source_version": "promoted_from_pr",  # Or fetch from DB if needed
        "gcs_path": gcs_path,
        "status": "production",
        "created_at": datetime.now().isoformat(),
    }

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"Error logging promotion to BQ: {errors}")
    else:
        print(f"Logged production status for {data_type} to BigQuery.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    client_gcs = storage.Client()
    bucket = client_gcs.bucket(args.bucket)
    project_id = client_gcs.project

    print(f"Promoting artifacts for PR {args.pr_number}...")

    # Define Paths
    paths = [
        (
            "raw",
            f"data/raw/pr-{args.pr_number}/raw.csv",
            "data/raw/production/latest.csv",
        ),
        (
            "processed",
            f"data/processed/pr-{args.pr_number}/processed.csv",
            "data/processed/production/latest.csv",
        ),
        (
            "features",
            f"data/features/pr-{args.pr_number}/features.csv",
            "data/features/production/features.csv",
        ),
    ]

    # 1. Promote Files & Log to BQ
    for data_type, source, dest in paths:
        # Copy File
        promote_blob(bucket, source, dest)

        # Log "Production" entry to BigQuery (Append Only)
        log_promotion_to_bq(
            project_id,
            int(args.pr_number),
            data_type,
            f"gs://{args.bucket}/{dest}",
        )


if __name__ == "__main__":
    main()
