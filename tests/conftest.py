# tests/conftest.py
import sys
import types
from pathlib import Path

# Ensure repo root is importable so "import src.*" works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -------------------- Mock google cloud SDKs --------------------
google = types.ModuleType("google")
google_cloud = types.ModuleType("google.cloud")
google_cloud_storage = types.ModuleType("google.cloud.storage")
google_cloud_bigquery = types.ModuleType("google.cloud.bigquery")


class DummyBlob:
    def exists(self):
        return False

    def upload_from_filename(self, *args, **kwargs):
        return None


class DummyBucket:
    def blob(self, *args, **kwargs):
        return DummyBlob()

    def copy_blob(self, *args, **kwargs):
        return None


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.project = "dummy-project"

    def bucket(self, *args, **kwargs):
        return DummyBucket()

    def insert_rows_json(self, *args, **kwargs):
        return []  # no errors

    def query(self, *args, **kwargs):
        class DummyJob:
            def result(self_inner):
                return []

        return DummyJob()


google_cloud_storage.Client = DummyClient
google_cloud_bigquery.Client = DummyClient

# Register into sys.modules BEFORE any src imports happen
sys.modules["google"] = google
sys.modules["google.cloud"] = google_cloud
sys.modules["google.cloud.storage"] = google_cloud_storage
sys.modules["google.cloud.bigquery"] = google_cloud_bigquery
