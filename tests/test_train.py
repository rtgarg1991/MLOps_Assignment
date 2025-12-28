# tests/test_train.py
import pandas as pd
from src.train import train_model, TARGET_COLUMN

def test_train_model_outputs():
    # 20 samples ensures stratified split works with test_size=0.2
    df = pd.DataFrame({
        "age": list(range(40, 60)),
        "sex": [0, 1] * 10,
        TARGET_COLUMN: [0, 1] * 10
    })

    config = {"model_type": "logistic_regression"}
    model, scaler, metrics, artifacts = train_model(df, config)

    assert model is not None
    assert scaler is not None
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert isinstance(artifacts, list)
    assert len(artifacts) > 0
