import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from google.cloud import storage

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


def run_eda(df: pd.DataFrame):
    print("Running EDA (before cleaning)...")

    artifacts = []

    # ---------------- Missing values ----------------
    missing = df.isna().sum()
    missing[missing > 0].plot(kind="bar", title="Missing Values")
    plt.tight_layout()

    file_path = VISUALS_DIR / "missing_values" / "missing_values.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)
    plt.close()

    artifacts.append((file_path, "missing_values.png"))

    # ---------------- Numeric distributions ----------------
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()

            file_path = VISUALS_DIR / "distributions" / f"{col}.png"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path)
            plt.close()

            artifacts.append((file_path, f"{col}.png"))

    # ---------------- Correlation heatmap ----------------
    corr = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    file_path = VISUALS_DIR / "correlations" / "correlation_heatmap.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)
    plt.close()

    artifacts.append((file_path, "correlation_heatmap.png"))

    return artifacts


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

    print(f"EDA Input: gs://{args.bucket}/{input_path}")

    # Read raw data
    df = pd.read_csv(f"gs://{args.bucket}/{input_path}")

    # Run EDA
    artifacts = run_eda(df)
    subpath = f"data/eda/pr-{args.pr_number}"

    # Upload EDA files
    for local_file, remote_name in artifacts:
        blob_path = f"{subpath}/{remote_name}"
        bucket.blob(blob_path).upload_from_filename(local_file)
        print(f" - Uploaded: {remote_name}")


if __name__ == "__main__":
    main()
