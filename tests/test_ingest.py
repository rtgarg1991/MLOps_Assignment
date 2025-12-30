# tests/test_ingest.py
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ingest import ingest_data


@patch("ucimlrepo.fetch_ucirepo")
def test_ingest_returns_dataframe(mock_fetch):
    fake = MagicMock()
    fake.data.features = pd.DataFrame({"age": [50, 60], "sex": [1, 0]})
    fake.data.targets = pd.DataFrame({"num": [0, 1]})
    mock_fetch.return_value = fake

    df = ingest_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "age" in df.columns
    assert "num" in df.columns
