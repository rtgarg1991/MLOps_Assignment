# tests/test_feature_engineering.py
import pandas as pd
from src.feature_engineering import one_hot_encode_features, TARGET_COLUMN


def test_one_hot_encoding():
    df = pd.DataFrame({"age": [50, 60], "sex": [0, 1], TARGET_COLUMN: [0, 1]})

    out = one_hot_encode_features(df, ["sex"])

    assert isinstance(out, pd.DataFrame)
    assert TARGET_COLUMN in out.columns
    assert "sex_0" in out.columns
    assert "sex_1" in out.columns
