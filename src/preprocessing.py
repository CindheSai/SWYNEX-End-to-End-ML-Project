from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Attrition"
IDENTIFIER_CANDIDATES = ["EmployeeNumber", "EmployeeID", "EmployeeId"]

def load_dataset(path):
    """Load and validate the employee attrition CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV in the project's data/ folder."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

def prepare_features(df):
    """Return feature matrix X and binary target y; remove identifiers and invalid labels."""
    data = df.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET].isin(["Yes", "No"])].copy()
    if data.empty:
        raise ValueError("No rows with valid Attrition labels ('Yes'/'No') remain.")
    y = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = data.drop(columns=[TARGET])
    identifiers = [c for c in IDENTIFIER_CANDIDATES if c in X.columns]
    X = X.drop(columns=identifiers)
    constants = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constants)
    return X, y

def build_preprocessor(X):
    """Create fit-inside-training-fold transformations for numeric and categorical features."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")
