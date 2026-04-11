# Credit Risk Modeling with LendingClub Data

End-to-end credit risk project covering **probability of default (PD)**, **loss given default (LGD)**, **exposure at default (EAD)**, **expected loss (EL)**, and **IFRS 9 style staging** using LendingClub consumer loan data.

The project is built as a six-notebook workflow with reusable Python modules for preprocessing, validation, and portfolio loss calculations.

## Project Highlights

| Component | Outcome |
|---|---|
| Dataset | 933,160 raw loans, 37 raw columns |
| PD model | XGBoost with rolling time-window validation |
| PD performance | Mean out-of-sample ROC-AUC `0.6174` |
| Validation | KS `0.189`, PSI `0.026` (Stable), SHAP explainability |
| Return model | Ridge regression, out-of-sample R² `-0.003` |
| LGD model | Ridge regression, out-of-sample R² `0.006` |
| Final scoring | 112,858 unseen loans scored |
| Portfolio EL on final scored set | `$45.3M` |

## What This Project Does

1. Cleans and explores LendingClub loan data.
2. Engineers borrower affordability, pricing, and credit-quality features.
3. Trains a PD classifier with out-of-time validation.
4. Validates model quality with credit-risk metrics:
   - ROC-AUC
   - KS statistic
   - PSI
   - threshold analysis
   - SHAP
5. Builds a portfolio risk layer using:
   - PD
   - LGD
   - EAD
   - Expected Loss
   - IFRS 9 style Stage 1/2/3 segmentation

## Notebook Workflow

| Notebook | Purpose | Key Output |
|---|---|---|
| [01_eda_and_cleaning.ipynb](notebooks/01_eda_and_cleaning.ipynb) | EDA, missing values, distributions, target analysis | Cleaned dataset understanding |
| [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) | Log, ratio, interaction, and composite features | Processed modeling dataset |
| [03_model_training.ipynb](notebooks/03_model_training.ipynb) | Rolling-window training for PD and return models | XGBoost PD model, Ridge return model, saved artifacts |
| [04_model_validation.ipynb](notebooks/04_model_validation.ipynb) | KS, PSI, threshold analysis, SHAP | Model validation and interpretability |
| [05_lgd_ead_el_ifrs9_stress.ipynb](notebooks/05_lgd_ead_el_ifrs9_stress.ipynb) | LGD, EAD, EL, IFRS 9 staging, stress testing | Portfolio loss framework |
| [06_final_predictions.ipynb](notebooks/06_final_predictions.ipynb) | Score unseen test loans | Final PD, LGD, EAD, EL, stage outputs |

## Dataset

| Item | Value |
|---|---|
| Source | [LendingClub on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Raw size | `933,160` loans |
| Time span | `2008-01-01` to `2016-12-01` |
| Targets | `early_default`, `return` |
| Missingness handled | `emp_length`, `mths_since_last_delinq` |

Raw data is intentionally excluded from version control. Place the Kaggle files below in `data/raw/`:

- `lc_loan.csv`
- `lc_loan_test.csv`

## Feature Engineering

The project uses raw variables plus engineered features across affordability, pricing, and credit quality.

| Category | Examples |
|---|---|
| Log features | `log_loan_amnt`, `log_dti`, `log_revol_bal`, `log_annual_inc` |
| Credit quality | `fico_avg`, `fico_int_rate_interaction`, `fico_int_rate_ratio` |
| Affordability | `loan_amnt_to_income_ratio`, `payment_to_income_ratio` |
| Interactions | `grade_dti_interaction`, `purpose_loan_amnt_interaction`, `installment_fico_interaction` |
| Relative features | `relative_int_rate`, `loan_to_purpose_amnt_ratio`, `composite_loan_feature` |

Preprocessing was made deterministic for scoring by:

- removing `issue_d` from one-hot encoding (prevented leakage of origination date into OHE features)
- using fixed mappings for `purpose_encoded` and `sub_grade_encoded` (eliminated `cat.codes` instability across datasets)
- persisting training reference tables for region- and purpose-based aggregates

## Modeling Approach

### PD Model

- Model: `XGBoost`
- Validation: rolling 36-month train / 12-month test windows
- Imbalance handling: `SMOTE`
- Threshold tuning: F1-optimized threshold from precision-recall curve

### Return Model

- Model: `Ridge Regression`
- Selected after removing LightGBM which showed consistently negative R² out-of-sample

### LGD / EAD / EL Layer

- LGD modeled on defaulted loans only
- EAD computed from outstanding principal
- Expected Loss calculated as:

```text
EL = PD × LGD × EAD
```

## Validated Model Results

### PD Model

| Metric | Value |
|---|---:|
| Mean out-of-sample ROC-AUC | `0.6174` |
| Final-window ROC-AUC | `0.6320` |
| KS statistic | `0.189` |
| PSI (train → test) | `0.026` (Stable) |
| F1-optimal threshold | `0.051` |

### Rolling Window AUC

| Window | ROC-AUC |
|---|---:|
| 0 | `0.6193` |
| 1 | `0.5899` |
| 2 | `0.6185` |
| 3 | `0.6271` |
| 4 | `0.6320` |

### Return Model

| Metric | Value |
|---|---:|
| In-sample R² | `0.038` |
| Out-of-sample R² | `-0.003` |

### LGD Model

| Metric | Value |
|---|---:|
| Defaulted loans used | `49,349` |
| Mean LGD | `74.70%` |
| Out-of-sample R² | `0.006` |
| Out-of-sample MAE | `0.091` |

## Portfolio Results (NB05 — Labeled Validation Window)

| Metric | Value |
|---|---:|
| Loans in test window | `282,138` |
| Mean PD | `4.29%` |
| Mean LGD | `75.15%` |
| Mean EAD | `$12,803` |
| Total EAD | `$3,612,321,825` |
| Total Expected Loss | `$116,347,162` |
| Portfolio EL Rate | `3.22%` |

### IFRS 9 Stage Distribution (NB05)

| Stage | Loans | EAD Share | EL Rate |
|---|---:|---:|---:|
| Stage 1 | `250,510` | `88.82%` | `1.78%` |
| Stage 2 | `27,428` | `9.69%` | `12.11%` |
| Stage 3 | `4,200` | `1.50%` | `31.32%` |

## Final Portfolio Scoring Results (NB06 — Unseen Loans)

| Metric | Value |
|---|---:|
| Loans scored | `112,858` |
| Mean predicted PD | `4.60%` |
| Mean predicted LGD | `75.22%` |
| Mean EAD | `$12,392` |
| Total portfolio EAD | `$1,398,481,550` |
| Total Expected Loss | `$45,276,697` |
| Portfolio EL Rate | `3.24%` |

### IFRS 9 Stage Distribution (NB06)

| Stage | Loans | Share |
|---|---:|---:|
| Stage 1 | `98,930` | `87.7%` |
| Stage 2 | `12,262` | `10.9%` |
| Stage 3 | `1,666` | `1.5%` |

## Key Implementation Notes

### Preprocessing fixes applied

During development, five preprocessing bugs from the original implementation were identified and corrected:

1. `issue_d` OHE — origination date was being one-hot encoded, causing all test loans to get zero-vectors for 36 date-specific features
2. `annual_income_squared` — small income shifts amplified to extreme values after squaring; replaced with `log_annual_inc`
3. `purpose_encoded` via `cat.codes` — integer mapping unstable across datasets; replaced with fixed dictionary
4. `sub_grade_encoded` via `cat.codes` — same issue; replaced with fixed dictionary
5. Dataset-relative features recomputed from test data — `relative_int_rate`, `loan_to_purpose_amnt_ratio` now anchored to training reference tables

### Inference pipeline fix

XGBoost was trained on dense arrays (SMOTE output), but final scoring initially passed sparse matrices from the preprocessor. Sparse structural zeros were treated as missing values by XGBoost, inflating predicted PDs to ~99%. Fixed by calling `.toarray()` before `predict_proba()`.

## Repo Structure

```text
credit-risk-modeling/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_validation.ipynb
│   ├── 05_lgd_ead_el_ifrs9_stress.ipynb
│   └── 06_final_predictions.ipynb
├── src/
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── lgd_ead.py
├── reports/
│   ├── credit_risk_modeling_report.tex
│   └── credit_risk_modeling_report.pdf
├── data/
│   ├── raw/         # local only, not tracked
│   ├── processed/   # local artifacts, not tracked
│   └── outputs/     # generated outputs, not tracked
└── models/          # trained artifacts, not tracked
```

## How To Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order:

1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_model_training.ipynb`
4. `notebooks/04_model_validation.ipynb`
5. `notebooks/05_lgd_ead_el_ifrs9_stress.ipynb`
6. `notebooks/06_final_predictions.ipynb`

## Tech Stack

`Python 3.11`, `pandas`, `NumPy`, `scikit-learn`, `XGBoost`, `imbalanced-learn`, `SHAP`, `Matplotlib`, `SciPy`, `joblib`

## Report

A full academic project report is included:

- [reports/credit_risk_modeling_report.pdf](reports/credit_risk_modeling_report.pdf)
- [reports/credit_risk_modeling_report.tex](reports/credit_risk_modeling_report.tex)
