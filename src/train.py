import argparse
import os
import pickle
from google.auth import default
from google.cloud import storage
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage


COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]


def load_data():
    path = "data/processed.cleveland.data"
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Replace missing values
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert to numeric
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col])

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    return df


def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = load_data()
    model, scaler = train_model(df)

    local_model_path = "/tmp/model.pkl"

    with open(local_model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    if args.model_dir.startswith("gs://"):
        parts = args.model_dir.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        # Handle cases where there might not be a subfolder path
        blob_path = parts[1] + "/model.pkl" if len(parts) > 1 else "model.pkl"

        print("\n====== GCP RUNTIME DEBUG ======")

        # What credentials are we using?
        creds, project = default()
        print("Default project from credentials:", project)
        print("Credential type:", type(creds))

        # What service account is this pod running as?
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                headers={"Metadata-Flavor": "Google"},
                timeout=2,
            )
            print("Pod service account:", r.text)
        except Exception as e:
            print("Metadata server error:", e)

        # What project does GCS client believe?
        client = storage.Client()
        print("GCS client project:", client.project)

        print("====== END DEBUG ======\n")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_model_path)

    print("Model saved successfully")


if __name__ == "__main__":
    main()
