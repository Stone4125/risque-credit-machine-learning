# Credit Risk Prediction

Binary classification pipeline that estimates whether a loan applicant will default. The study lives in one notebook: exploration, cleaning, a leakage-free preprocessing pipeline, a comparison of four models, tuning, a single test evaluation, and a prediction loaded back from disk.

## Dataset

Public Kaggle credit-risk table at `data/archive/credit_risk_dataset.csv` (the original download is also kept as `data/archive.zip`).

| Item | Value |
| --- | --- |
| Rows | 32,581 applicants |
| Target | `loan_status` (1 = default, 0 = no default) |
| Raw default rate | 21.82% |
| Missing values | `person_emp_length` (895), `loan_int_rate` (3,116) |

Predictors cover the applicant (`person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `cb_person_default_on_file`, `cb_person_cred_hist_length`) and the loan (`loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`).

`loan_grade` and `loan_int_rate` are assigned by the lender. They already carry part of the original credit decision, so the model reproduces that kind of score. It is not a decision made before the grade is known.

## Project layout

```
Projet_Risque_Credit/
├── data/
│   ├── archive.zip
│   └── archive/credit_risk_dataset.csv
├── images/                         # charts written by the notebook
├── models/
│   └── modele_risque_credit_final.pkl
├── notebooks/
│   └── analyse_risque_credit.ipynb
├── src/risque_credit/
│   ├── __init__.py
│   └── preprocessing.py            # grade-wise imputation, fit on train only
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Pipeline

`notebooks/analyse_risque_credit.ipynb` runs in this order:

1. Load the CSV through a path relative to the project root.
2. Explore missing values, class balance, numeric correlations, age, and default rates by loan grade, home ownership, and loan purpose.
3. Drop impossible rows only: age outside 18–100, employment length above 60 years or longer than the applicant's age. Missing values stay in the table.
4. Hold out 20% as the test set before any learned statistic. Split the training rows again into a fit set and a validation set.
5. For each model, fit an `imbalanced-learn` pipeline on the fit set only:
   - impute interest rate with the training median of the same loan grade, and employment length with the training median
   - scale numeric columns and one-hot encode categories
   - apply SMOTE only while fitting
   - train the classifier
6. Rank logistic regression, random forest, SVM, and XGBoost by validation ROC AUC. The validation and test sets keep the real default rate, because SMOTE is skipped at prediction time.
7. Tune the selected model with a 3-fold stratified grid search on the full training set (`scoring="roc_auc"`).
8. Score that tuned pipeline once on the untouched test set: classification report, F1, precision, recall, ROC AUC, Gini, KS, confusion matrix, ROC curve, and a calibration curve.
9. Save the whole pipeline to `models/modele_risque_credit_final.pkl` and score one fictional applicant from the reloaded file.

## Setup

Python 3.11 or newer. Pinned libraries are in `requirements.txt`.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The editable install is required. The saved pipeline includes `ImputationParGroupe` from this project, and `joblib` has to import that class when the model is loaded or when grid search uses several processes.

## Run

From the project root:

```bash
jupyter notebook notebooks/analyse_risque_credit.ipynb
```

Run all cells from top to bottom. Charts are written to `images/`. The fitted pipeline is written to `models/`.

The comparison cell trains an RBF SVM, and the following cell runs a grid search. Expect that part to take several minutes.

## Results

Seven impossible rows were removed (32,581 → 32,574). The test set has 6,515 real applicants and a 21.81% default rate.

Untuned models, scored on the validation split (not the test set):

| Model | F1 (default) | ROC AUC | Recall (default) |
| --- | --- | --- | --- |
| Logistic regression | 0.643 | 0.878 | 0.786 |
| Random forest | 0.826 | 0.934 | 0.740 |
| SVM | 0.759 | 0.916 | 0.766 |
| XGBoost | 0.828 | 0.945 | 0.738 |

XGBoost is selected on validation ROC AUC. The grid search keeps `learning_rate=0.1`, `max_depth=7`, `n_estimators=200` (mean cross-validated ROC AUC 0.940).

Tuned XGBoost on the untouched test set:

| Metric | Value |
| --- | --- |
| F1 (default) | 0.824 |
| Precision (default) | 0.966 |
| Recall (default) | 0.719 |
| ROC AUC | 0.943 |
| Gini | 0.886 |
| KS | 0.736 |

The fictional applicant (age 34, income 42,000, renter, medical loan of 8,500, grade C) is classified as non-default, with an estimated default probability of 7.57% from the model reloaded from disk.

Re-run the notebook if you change the data or the split: these figures belong to `random_state=42`.

## Limitations

- Grade and interest rate are lender-assigned features, as noted above.
- SMOTE runs after one-hot encoding, so synthetic rows can contain fractional category codes. It still does not touch the test set.
- The calibration curve shows whether announced probabilities match observed default frequencies. The model is not recalibrated after that check.
- There is one hold-out test and no cost matrix for a lending decision. Threshold 0.5 is the default class cutoff, not an optimized acceptance policy.
- This is a study pipeline, not a production credit score.
