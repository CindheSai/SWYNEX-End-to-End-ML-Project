import pandas as pd
from src.preprocessing import prepare_features, build_preprocessor

def test_prepare_features_encodes_target_and_drops_identifier():
    df = pd.DataFrame({
        "Attrition": ["No", "Yes", "No"],
        "EmployeeNumber": [101, 102, 103],
        "Age": [30, 40, 35],
        "Department": ["Sales", "HR", "Sales"],
        "Constant": [1, 1, 1]
    })
    X, y = prepare_features(df)
    assert list(y) == [0, 1, 0]
    assert "EmployeeNumber" not in X.columns
    assert "Constant" not in X.columns
    assert set(X.columns) == {"Age", "Department"}

def test_preprocessor_fits_and_transforms_mixed_data():
    X = pd.DataFrame({
        "Age": [20, 30, None, 40],
        "Department": ["Sales", "HR", "Sales", None]
    })
    prep = build_preprocessor(X)
    transformed = prep.fit_transform(X)
    assert transformed.shape[0] == 4
    assert transformed.shape[1] >= 2
