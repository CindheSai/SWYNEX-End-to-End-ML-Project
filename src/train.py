import json
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from src.preprocessing import load_dataset, prepare_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = ROOT / "models" / "employee_attrition_model.joblib"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RANDOM_STATE = 42

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    fitted, rows = {}, []
    sns.set_theme(style="whitegrid")
    for name, classifier in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=[0, 1],
            display_labels=["No Attrition", "Attrition"], values_format="d", ax=ax
        )
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        fig.savefig(FIGURES / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=160)
        plt.close(fig)
        print(f"\n{name}\n{classification_report(y_test, pred, target_names=['No Attrition','Attrition'], zero_division=0)}")

    comparison = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))

    tune_pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])
    grid = GridSearchCV(
        tune_pipe,
        {"classifier__C": [0.001, 0.01, 0.1, 1, 10],
         "classifier__class_weight": [None, "balanced"]},
        scoring="recall",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    final_model = grid.best_estimator_
    pred = final_model.predict(X_test)
    prob = final_model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "Tuned Logistic Regression",
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "best_parameters": grid.best_params_,
        "best_mean_cross_validation_recall": float(grid.best_score_),
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "features_before_encoding": int(X.shape[1])
    }
    joblib.dump(final_model, MODEL_PATH)
    (RESULTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
    ax.set_title("ROC Curve — Tuned Logistic Regression")
    fig.tight_layout()
    fig.savefig(FIGURES / "tuned_logistic_regression_roc_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
        kind="bar", ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Employee Attrition Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=160)
    plt.close(fig)
    print("\nFinal tuned model metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved results and figures under: {RESULTS}")

if __name__ == "__main__":
    main()
