import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "employee_attrition_model.joblib"

def main():
    parser = argparse.ArgumentParser(description="Predict employee attrition from a CSV of employee records.")
    parser.add_argument("--input", required=True, help="CSV containing employee feature columns (without Attrition target).")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a trained .joblib pipeline.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `python -m src.train` first.")
    records = pd.read_csv(args.input)
    if "Attrition" in records.columns:
        records = records.drop(columns=["Attrition"])
    # Match training preprocessing: identifier fields are not model features.
    records = records.drop(columns=[c for c in ["EmployeeNumber", "EmployeeID", "EmployeeId"] if c in records.columns])
    model = joblib.load(model_path)
    labels = model.predict(records)
    probabilities = model.predict_proba(records)[:, 1]
    result = records.copy()
    result["predicted_attrition"] = ["Yes" if int(v) == 1 else "No" for v in labels]
    result["attrition_probability"] = probabilities.round(4)
    result.to_csv(args.output, index=False)
    print(result[["predicted_attrition", "attrition_probability"]].to_string(index=False))
    print(f"\nPredictions saved to {args.output}")

if __name__ == "__main__":
    main()
