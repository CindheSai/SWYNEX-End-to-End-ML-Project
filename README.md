# SWYNEX — End-to-End Employee Attrition Prediction

A portfolio-ready educational machine-learning project using the IBM-style HR Employee Attrition dataset. It demonstrates data validation, leakage-aware preprocessing, model comparison, hyperparameter tuning, evaluation, model serialization, and batch prediction.

> **Responsible use:** This project is for learning and demonstration. Attrition predictions are uncertain and must not be used as the sole basis for employment, promotion, termination, compensation, or other decisions affecting employees.

## Dataset

- File: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Observations: 1,470
- Columns: 35
- Target: `Attrition` (`Yes` / `No`)
- Target distribution in the supplied file: 1,233 `No`, 237 `Yes`
- `EmployeeNumber` is an identifier and is excluded from the model features.

The supplied CSV is included for local reproducibility. Do not publish raw data to GitHub unless you have verified its license and your organization's data-sharing rules. The `.gitignore` excludes `data/*.csv` by default.

## Project structure

```text
SWYNEX-End-to-End-ML-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── SWYNEX_End_to_End_Employee_Attrition.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   ├── README.md
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   └── README.md
├── results/
│   ├── model_comparison.csv
│   ├── final_metrics.json
│   └── figures/
└── tests/
    └── test_preprocessing.py
```

Generated model and results files appear after training.

## Setup (Windows PowerShell)

1. Install Python 3.10 or newer.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

Run from the **main project folder**:

```powershell
python -m src.train
```

The script:
1. Loads and validates the CSV.
2. Removes rows with unknown target labels, identifier columns, and constant features.
3. Creates a stratified 80/20 train/test split.
4. Fits numeric and categorical preprocessing inside scikit-learn pipelines.
5. Trains Logistic Regression and Random Forest baselines.
6. Compares accuracy, precision, recall, F1-score, and ROC-AUC.
7. Tunes Logistic Regression using 5-fold stratified cross-validation, optimizing recall.
8. Evaluates the selected model on the held-out test set and saves artifacts.

Outputs:
- `models/employee_attrition_model.joblib`
- `results/model_comparison.csv`
- `results/final_metrics.json`
- `results/figures/` (comparison chart, confusion matrices, ROC curve)

Metrics are computed when you run the project and may differ from results in earlier notebook sessions or on other library versions. Do not copy metrics into a report until the current run completes successfully.

## Use the trained model for predictions

Create a CSV containing employee feature columns, with the same feature names as the training dataset. The `Attrition` target column is optional and will be dropped if present.

```powershell
python -m src.predict --input path\to\new_employees.csv --output predictions.csv
```

The output includes `predicted_attrition` and `attrition_probability`. The probability is a model score, not a guarantee that an employee will leave.

## Run tests

```powershell
python -m pytest -q
```

## Google Colab notebook

Open `notebooks/SWYNEX_End_to_End_Employee_Attrition.ipynb` in Google Colab. Upload the supplied CSV when prompted and run the notebook from top to bottom. The final notebook cell downloads a ZIP of generated model and results artifacts. Keep the notebook's output files in the matching `models/` and `results/` folders.

## Methodology and limitations

- The target is encoded as `No = 0`, `Yes = 1`.
- Numeric values are median-imputed and standardized; categorical values are mode-imputed and one-hot encoded.
- Preprocessing is fitted as part of the model pipeline to reduce train/test leakage.
- Stratification preserves the target class ratio in train/test splits.
- Recall is used as the grid-search objective because the minority `Yes` class matters for this demonstration. Precision and false positives still matter, so review all metrics together.
- This dataset may not represent a real organization's workforce. The model may encode historical bias and should not be used to make consequential employee decisions.

## Internship submission checklist

- [ ] Run the notebook end-to-end.
- [ ] Run `python -m src.train` and confirm model, metrics, and figures were created.
- [ ] Run tests.
- [ ] Review the README and add only verified current-run metrics.
- [ ] Push the repository to GitHub using a repository name containing `SWYNEX`.
- [ ] Add a LinkedIn post explaining the workflow, model comparison, limitations, and lessons learned.
- [ ] Submit both the GitHub repository URL and LinkedIn post URL in the internship portal.
