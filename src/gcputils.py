from pathlib import Path
import requests

from google.cloud import storage
from google.auth import default


def gcp_runtime_debug():
    """
    Print runtime GCP authentication and environment details.
    """
    print("\n====== GCP RUNTIME DEBUG ======")

    creds, project = default()
    print("Default project from credentials:", project)
    print("Credential type:", type(creds))

    try:
        r = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
            timeout=2,
        )
        print("Service account:", r.text)
    except Exception as e:
        print("Metadata server error:", e)

    client = storage.Client()
    print("GCS client project:", client.project)

    print("====== END DEBUG ======\n")


def upload_file_to_gcs(
    local_file_path: Path,
    bucket_name: str,
    blob_path: str,
    debug: bool = True
):
    """
    Upload a single file to GCS.
    """
    if debug:
        gcp_runtime_debug()

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(str(local_file_path))
    print(f"Uploaded → gs://{bucket_name}/{blob_path}")


def upload_directory_to_gcs(
    local_dir: Path,
    bucket_name: str,
    gcs_prefix: str
):
    """
    Recursively upload a local directory to GCS.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            blob_path = f"{gcs_prefix}/{relative_path}"
            bucket.blob(blob_path).upload_from_filename(str(file_path))

    print(f"Directory uploaded → gs://{bucket_name}/{gcs_prefix}")
