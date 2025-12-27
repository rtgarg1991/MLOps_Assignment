import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import joblib

from gcputils import upload_directory_to_gcs, upload_file_to_gcs

from sklearn.model_selection import train_test_split
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

# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"
APP_MODELS_DIR = PROJECT_ROOT / "app" / "models"
CONFIG_PATH = PROJECT_ROOT / "train_config.yaml"

FEATURE_DATA_PATH = DATA_DIR / "heart_disease_features.csv"
TARGET_COLUMN = "disease_present"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)
APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = VISUALS_DIR / "metrics.csv"

# ================== GCP CONFIG ==================
GCP_BUCKET_NAME = DATA_DIR.name
GCP_ARTIFACT_PREFIX = "artifacts/heart_disease"

# ================== CONFIG LOADER ==================
def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {path}\n"
            f"Resolved PROJECT_ROOT: {PROJECT_ROOT}"
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)

# ================== SPLIT ==================
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

# ================== MODEL FACTORY ==================
def build_model(algo_name: str, params: Dict[str, Any]):
    if algo_name == "logistic_regression":
        return LogisticRegression(**params)
    elif algo_name == "random_forest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

# ================== EVALUATION ==================
def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    cm = confusion_matrix(y_test, model.predict(X_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close(fig)

def plot_precision_recall(model, X_test, y_test, model_name: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve - {model_name}")
    ax.grid(alpha=0.3)
    fig.savefig(VISUALS_DIR / f"precision_recall_{model_name}.png", dpi=300)
    plt.close(fig)

def plot_roc_curve(model, X_test, y_test, model_name: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(VISUALS_DIR / f"roc_{model_name}.png", dpi=300)
    plt.close()

def save_classification_report(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)

    print(f"\nClassification Report – {model_name}")
    print(classification_report(y_test, y_pred, zero_division=0))

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        VISUALS_DIR / f"classification_report_{model_name}.csv"
    )

# ================== MAIN ==================
def main():
    config = load_config(CONFIG_PATH)

    algo_name = config["algorithm"]["name"]
    params = config["parameters"][algo_name]
    is_experimental = config["isExperimental"]

    print(f"Algorithm: {algo_name}")
    print(f"Parameters: {params}")
    print(f"Experimental mode: {is_experimental}")

    df = pd.read_csv(FEATURE_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    splits = split_data(X, y, is_experimental, params.get("random_state", 42))

    model = build_model(algo_name, params)
    model.fit(splits["X_train"], splits["y_train"])

    model_path = APP_MODELS_DIR / f"{algo_name}_model.pkl"
    joblib.dump(model, model_path)

    metrics = evaluate_model(model, splits["X_test"], splits["y_test"], algo_name)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    plot_confusion_matrix(model, splits["X_test"], splits["y_test"], algo_name)
    plot_precision_recall(model, splits["X_test"], splits["y_test"], algo_name)
    plot_roc_curve(model, splits["X_test"], splits["y_test"], algo_name)
    save_classification_report(model, splits["X_test"], splits["y_test"], algo_name)

    print("\nFinal Metrics")
    print(metrics_df)

    # ---------- UPLOAD TO GCP ----------
    upload_directory_to_gcs(
        local_dir=VISUALS_DIR,
        bucket_name=GCP_BUCKET_NAME,
        gcs_prefix=f"{GCP_ARTIFACT_PREFIX}/training/visuals"
    )

    upload_directory_to_gcs(
        local_dir=APP_MODELS_DIR,
        bucket_name=GCP_BUCKET_NAME,
        gcs_prefix=f"{GCP_ARTIFACT_PREFIX}/training/models"
    )

    upload_file_to_gcs(
        local_file_path=metrics_path,
        bucket_name=GCP_BUCKET_NAME,
        blob_path=f"{GCP_ARTIFACT_PREFIX}/training/metrics/metrics.csv"
    )

if __name__ == "__main__":
    main()
