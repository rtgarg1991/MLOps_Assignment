# tests/test_train.py
import sys
from unittest.mock import MagicMock
import pandas as pd

# Mock mlflow BEFORE importing src.train
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()

from src.train import train_model  


def test_train_model_outputs():
    df = pd.DataFrame(
        {
            "age": [50, 60, 55, 45, 52, 61],
            "sex": [0, 1, 0, 1, 0, 1],
            "disease_present": [0, 1, 0, 1, 0, 1],
        }
    )

    config = {"model_type": "logistic_regression"}

    result = train_model(df, config)
    model, scaler, metrics, artifacts = result[:4]

    assert model is not None
    assert scaler is not None
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert isinstance(artifacts, list)


