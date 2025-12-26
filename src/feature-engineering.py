import pandas as pd
from pathlib import Path
from typing import List, Tuple

# ================== PATHS ==================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"
FEATURE_DATA_PATH = DATA_DIR / "heart_disease_features.csv"
FEATURE_LIST_PATH = DATA_DIR / "feature_columns.txt"

# ================== CONFIG ==================
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "disease_present"

# ================== FEATURE ENGINEERING ==================
def one_hot_encode_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical features and retain target column.

    Args:
        df: Cleaned dataframe (includes target column)
        categorical_cols: List of categorical feature names

    Returns:
        Tuple of:
        - Feature dataframe INCLUDING target column
        - List of feature column names (excluding target)
    """
    print("One-hot encoding categorical features...")

    # Separate columns
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in categorical_cols if c in df.columns]

    df_numeric = df[numeric_cols].copy()

    if cat_cols:
        df_cat = df[cat_cols].astype(str)
        df_cat_encoded = pd.get_dummies(df_cat, prefix=cat_cols, drop_first=False)
        df_features = pd.concat([df_numeric, df_cat_encoded], axis=1)
    else:
        df_features = df_numeric

    feature_columns = list(df_features.columns)

    # Add target column
    df_final = pd.concat(
        [df_features, df[[TARGET_COLUMN]]],
        axis=1
    )

    print(f"  Total features: {len(feature_columns)}")
    print(f"  Final dataset shape: {df_final.shape}")

    return df_final, feature_columns

# ================== MAIN ==================
def main():
    print(f"Loading cleaned data from {CLEAN_DATA_PATH}")
    df_clean = pd.read_csv(CLEAN_DATA_PATH)

    df_features, feature_cols = one_hot_encode_features(
        df_clean,
        CATEGORICAL_FEATURES
    )

    # Saving feature dataset
    df_features.to_csv(FEATURE_DATA_PATH, index=False)
    print(f"Feature dataset saved to {FEATURE_DATA_PATH}")

    # Saving feature column names
    with open(FEATURE_LIST_PATH, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print(f"Feature column list saved to {FEATURE_LIST_PATH}")

# ================== MAIN ==================
if __name__ == "__main__":
    main()
