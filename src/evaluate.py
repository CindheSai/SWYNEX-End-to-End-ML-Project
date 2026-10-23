"""Evaluate an existing saved model on a labelled CSV dataset."""
import json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.preprocessing import load_dataset, prepare_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "employee_attrition_model.joblib"))
    args = parser.parse_args()
    df = load_dataset(args.data)
    X, y = prepare_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = joblib.load(args.model)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
