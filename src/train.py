import argparse
import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from google.auth import default
from google.cloud import storage, bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

def load_data(bucket_name, pr_number):
    client_gcs = storage.Client()
    bucket = client_gcs.bucket(bucket_name)

    # LOGIC GATE: Use PR Features if available, else Production Features
    pr_feat_path = f"data/features/pr-{pr_number}/features.csv"
    prod_feat_path = "data/features/production/features.csv"
    
    input_path = pr_feat_path if bucket.blob(pr_feat_path).exists() else prod_feat_path
    input_uri = f"gs://{bucket_name}/{input_path}"
    print(f"Training loading features from: {input_uri}")
    
    return pd.read_csv(input_uri)

def train_model(df, config):
    """Trains a model based on the provided configuration dictionary."""
    X = df.drop("target", axis=1)
    y = df["target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 1. Choose Model Type
    if config['model_type'] == 'logistic_regression':
        model = LogisticRegression(max_iter=config.get('max_iter', 1000))
    elif config['model_type'] == 'random_forest':
        model = RandomForestClassifier(n_estimators=config.get('n_estimators', 100))
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # 2. Generate Artifacts (Confusion Matrix)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.title(f"Confusion Matrix: {config['model_type']}")
    cm_path = "/tmp/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, output_dict=True)
    }
    
    # List of (local_path, destination_filename)
    artifacts = [(cm_path, "confusion_matrix.png")]
    
    return model, scaler, metrics, artifacts

def log_to_bigquery(project_id, table_id, row):
    client = bigquery.Client(project=project_id)
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"BigQuery Insert Errors: {errors}")
    else:
        print("Logged to BigQuery successfully.")

def fetch_best_config(project_id, pr_number):
    client = bigquery.Client(project=project_id)
    # LOGIC UPDATE: Select best accuracy ONLY from the latest run_id for this PR
    query = f"""
        SELECT config_params 
        FROM `{project_id}.ml_metadata.experiments`
        WHERE pr_number = {pr_number}
        AND run_id = (
            SELECT MAX(run_id) 
            FROM `{project_id}.ml_metadata.experiments` 
            WHERE pr_number = {pr_number}
        )
        ORDER BY accuracy DESC
        LIMIT 1
    """
    query_job = client.query(query)
    results = list(query_job.result())
    
    if not results:
        print(f"No experiments found for PR {pr_number}.")
        return None
    
    import ast
    return ast.literal_eval(results[0].config_params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", choices=['experiment', 'full_training'], required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--model-type", choices=['logistic_regression', 'random_forest'], default='logistic_regression')
    args = parser.parse_args()

    project_id = bigquery.Client().project
    bucket_name = args.model_dir.replace("gs://", "").split("/")[0]
    
    df = load_data(bucket_name, args.pr_number)

    if args.mode == 'full_training':
        print(f"Production Training: Fetching best config for PR {args.pr_number}")
        best_config = fetch_best_config(project_id, args.pr_number)
        config = best_config if best_config else {"model_type": args.model_type}
    else:
        config = {"model_type": args.model_type}

    model, scaler, metrics, artifacts = train_model(df, config)
    
    # Save Model Locally
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    local_model_path = "/tmp/model.pkl"
    with open(local_model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "config": config}, f)

    client_gcs = storage.Client(project=project_id)
    bucket = client_gcs.bucket(bucket_name)

    # Determine Destination Paths
    if args.mode == 'experiment':
        subpath = f"experiments/pr-{args.pr_number}/{timestamp}"
        tracking_row = {
            "run_id": timestamp, "timestamp": datetime.now().isoformat(),
            "pr_number": args.pr_number, "model_type": config["model_type"],
            "accuracy": metrics["accuracy"], "gcs_path": f"gs://{bucket_name}/{subpath}",
            "config_params": str(config)
        }
        log_to_bigquery(project_id, f"{project_id}.ml_metadata.experiments", tracking_row)
    else:
        # PRODUCTION MODE
        subpath = f"production/model_v_{args.pr_number}"
        prod_row = {
            "model_id": f"heart-model-v{args.pr_number}",
            "version_tag": str(args.pr_number),
            "deployment_date": datetime.now().isoformat(),
            "is_active": True,
            "performance_at_deploy": metrics["accuracy"]
        }
        log_to_bigquery(project_id, f"{project_id}.ml_metadata.production_models", prod_row)

    # --- UPLOAD MODEL & ARTIFACTS ---
    print(f"Uploading artifacts to {subpath}...")
    
    # 1. Model
    bucket.blob(f"{subpath}/model.pkl").upload_from_filename(local_model_path)
    
    # 2. Images (Confusion Matrix, etc.)
    for local_file, remote_name in artifacts:
        blob_path = f"{subpath}/{remote_name}"
        bucket.blob(blob_path).upload_from_filename(local_file)
        print(f" - Uploaded: {remote_name}")

    # --- ADDED: PRINT METRICS FOR GITHUB ACTIONS LOG SCRAPER ---
    # This allows the runner to see the metrics without needing BigQuery permissions
    final_metrics = {
        "run_id": timestamp,
        "model_type": config["model_type"],
        "accuracy": metrics["accuracy"],
        "pr_number": args.pr_number
    }
    print(f"__METRICS__:{json.dumps(final_metrics)}")

if __name__ == "__main__":
    main()