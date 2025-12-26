import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import joblib

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
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
VISUALS_DIR = PROJECT_ROOT / "visuals"

FEATURE_DATA_PATH = DATA_DIR / "heart_disease_features.csv"
TARGET_COLUMN = "disease_present"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)
APP_MODELS_DIR = PROJECT_ROOT / "app" / "models"
APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================== SPLIT ==================
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:

    print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

# ================== TRAIN ==================
def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:

    print("Training models...")

    lr_model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state
    )
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    print("Models trained successfully")

    return {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }
    
def save_models(models: Dict[str, Any]) -> None:
    """
    Persist trained models to app/models directory.
    """
    print("Saving trained models...")

    for model_name, model in models.items():
        filename = model_name.lower().replace(" ", "_") + "_model.pkl"
        model_path = APP_MODELS_DIR / filename
        joblib.dump(model, model_path)
        print(f"  Saved {model_name} → {model_path}")

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
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    path = VISUALS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=300)
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

    path = VISUALS_DIR / f"precision_recall_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    path = VISUALS_DIR / "roc_curves.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_classification_report(models, X_test, y_test):
    reports = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        print(f"\nClassification Report – {model_name}")
        print(classification_report(y_test, y_pred, zero_division=0))

        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report_dict).transpose()
        df_report["Model"] = model_name
        reports.append(df_report)

    final_report = pd.concat(reports)
    final_report.to_csv(VISUALS_DIR / "classification_report.csv")

# ================== MAIN ==================
def main():
    df = pd.read_csv(FEATURE_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    splits = split_data(X, y)
    models = train_models(splits["X_train"], splits["y_train"])

    metrics_list = []
    for model_name, model in models.items():
        metrics_list.append(
            evaluate_model(model, splits["X_test"], splits["y_test"], model_name)
        )
        plot_confusion_matrix(model, splits["X_test"], splits["y_test"], model_name)
        plot_precision_recall(model, splits["X_test"], splits["y_test"], model_name)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(VISUALS_DIR / "metrics.csv", index=False)

    plot_roc_curves(models, splits["X_test"], splits["y_test"])
    save_classification_report(models, splits["X_test"], splits["y_test"])
    save_models(models)

    print("\nFinal Metrics:")
    print(metrics_df)

if __name__ == "__main__":
    main()
