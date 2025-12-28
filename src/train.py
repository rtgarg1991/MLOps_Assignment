import argparse
from pathlib import Path
from typing import Dict, Any
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from google.cloud import storage, bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    classification_report
)

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUALS_DIR = PROJECT_ROOT / "visuals"
metrics_path = VISUALS_DIR / "metrics.csv"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COLUMN = "disease_present"

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

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    is_experimental: bool,
    random_state: int
) -> Dict[str, Any]:

    if is_experimental:
        print("Experimental mode → using only 80% data (60% train, 20% test)")

        # Drop 20% data
        X_used, _, y_used, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )

        # Split remaining 80% → 60/20
        X_train, X_test, y_train, y_test = train_test_split(
            X_used,
            y_used,
            test_size=0.25,   # 20 / 80
            stratify=y_used,
            random_state=random_state
        )

    else:
        print("Standard mode → using full data (80% train, 20% test)")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def train_model(df, config):
    """Trains a model based on the provided configuration dictionary."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    scaler = StandardScaler()
        
    splits =  split_data(X, y, False, 42)

    algo_name = config['model_type'] 
    # 1. Choose Model Type
    if algo_name == 'logistic_regression':
        model = LogisticRegression(max_iter=config.get('max_iter', 1000))
    elif algo_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=config.get('n_estimators', 100))
    else:
        raise ValueError(f"Unsupported model type: {algo_name}")
    
    #Train the model
    model.fit(splits["X_train"], splits["y_train"])

    
    metrics = evaluate_model(model, splits["X_test"], splits["y_test"], algo_name)
    metrics_df = pd.DataFrame([metrics])
    metrics_df["timestamp"] = datetime.now().isoformat()
    metrics_df.to_csv(metrics_path, mode="a",header=not metrics_path.exists(),index=False)


    cm_path = plot_confusion_matrix(model, splits["X_test"], splits["y_test"], algo_name)
    precision_recall_path = plot_precision_recall(model, splits["X_test"], splits["y_test"], algo_name)
    roc_curve_path = plot_roc_curve(model, splits["X_test"], splits["y_test"], algo_name)
    classification_report_path = save_classification_report(model, splits["X_test"], splits["y_test"], algo_name)

    
    # List of (local_path, destination_filename)
    artifacts = [(cm_path, "confusion_matrix.png"), (precision_recall_path, "precision_recall_.png"), 
                 (roc_curve_path, "roc_curve.png"), (classification_report_path, "classification_report.csv")]
    
    return model, scaler, metrics, artifacts

# ================== EVALUATION ==================
def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None
    }

def plot_confusion_matrix(model, X_test, y_test, model_name: str)-> Path:
    cm = confusion_matrix(y_test, model.predict(X_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    file_path = VISUALS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(file_path, dpi=300)
    plt.close(fig)
    return file_path

def plot_precision_recall(model, X_test, y_test, model_name: str)-> Path:
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    precision_recall_path = VISUALS_DIR / f"precision_recall_{model_name}.png" 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve - {model_name}")
    ax.grid(alpha=0.3)
    fig.savefig(precision_recall_path, dpi=300)
    plt.close(fig)
    return precision_recall_path
    

def plot_roc_curve(model, X_test, y_test, model_name: str)-> Path:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    roc_path = VISUALS_DIR / f"roc_{model_name}.png"

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(roc_path, dpi=300)
    plt.close()
    return roc_path

def save_classification_report(model, X_test, y_test, model_name: str)-> Path:
    y_pred = model.predict(X_test)
    classification_report_path = VISUALS_DIR / f"classification_report_{model_name}.csv"
    print(f"\nClassification Report – {model_name}")
    print(classification_report(y_test, y_pred, zero_division=0))

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        classification_report_path
    )
    
    return classification_report_path

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
            "run_id": timestamp,
            "timestamp": datetime.now().isoformat(),
            "pr_number": args.pr_number,
            "model_type": metrics["model"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "gcs_path": f"gs://{bucket_name}/{subpath}",
            "config_params": json.dumps(config),
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
            "accuracy_at_deploy": metrics["accuracy"],
            "roc_auc_at_deploy": metrics["roc_auc"],
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
        "model_type": metrics["model"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_number": args.pr_number,
    }

    print(f"__METRICS__:{json.dumps(final_metrics)}")

if __name__ == "__main__":
    main()