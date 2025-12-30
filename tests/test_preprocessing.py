# tests/test_preprocessing.py
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

# Dynamically import src/pre-processing.py (hyphenated filename can't be imported normally)
SRC_FILE = Path(__file__).resolve().parents[1] / "src" / "pre-processing.py"

spec = importlib.util.spec_from_file_location("pre_processing_module", SRC_FILE)
pre_processing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pre_processing)

clean_data = pre_processing.clean_data


def test_clean_data_converts_question_marks_to_nan():
    df = pd.DataFrame({"age": [50, "?"], "sex": [1, 0], "cp": [3, 2], "num": [0, 1]})

    cleaned = clean_data(df)

    # "?" should become NaN for numeric columns
    assert np.isnan(cleaned.loc[1, "age"])
    # Derived binary label should exist
    assert "disease_present" in cleaned.columns


def test_target_created():
    df = pd.DataFrame({"num": [0, 2, 1]})
    cleaned = clean_data(df)

    assert "disease_present" in cleaned.columns
    assert cleaned["disease_present"].tolist() == [0, 1, 1]
