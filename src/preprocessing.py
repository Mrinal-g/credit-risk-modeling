"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the
Credit Risk Modeling project (LendingClub dataset).

Usage:
    from src.preprocessing import (
        handling_data, create_features,
        fit_reference_tables, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
        PURPOSE_ENCODING, SUB_GRADE_ENCODING,
    )
"""

import re
import numpy as np
import pandas as pd


# ── Feature lists ──────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = list(set([
    # Log-transformed features
    'log_loan_amnt', 'log_dti', 'log_revol_bal', 'log_annual_inc',
    # Credit score
    'fico_avg', 'annual_inc',
    # Interaction terms
    'dti_income_interaction', 'fico_int_rate_interaction', 'revol_loan_interaction',
    'grade_interest_interaction', 'fico_loan_interaction',
    'delinq_int_rate_interaction', 'delinq_squared', 'grade_dti_interaction',
    'purpose_loan_amnt_interaction', 'installment_fico_interaction',
    # Ratios and relative features
    'loan_amnt_to_income_ratio', 'payment_to_income_ratio', 'revol_util_income_interaction',
    'fico_int_rate_ratio', 'int_rate_squared', 'relative_int_rate',
    'loan_to_purpose_amnt_ratio', 'composite_loan_feature',
]))

CATEGORICAL_FEATURES = [
    'sub_grade_encoded', 'home_ownership', 'verification_status',
    'earliest_cr_line', 'purpose_encoded', 'zip_region',
]

# ── Fixed categorical encodings (stable across all datasets) ──────────────────
# Derived from the full training corpus (lc_loan.csv, alphabetical sort).
# Any unseen category maps to -1 (treated as unknown by the OHE pipeline).

PURPOSE_ENCODING: dict[str, int] = {
    'car': 0,
    'credit_card': 1,
    'debt_consolidation': 2,
    'educational': 3,
    'home_improvement': 4,
    'house': 5,
    'major_purchase': 6,
    'medical': 7,
    'moving': 8,
    'other': 9,
    'renewable_energy': 10,
    'small_business': 11,
    'vacation': 12,
    'wedding': 13,
}

SUB_GRADE_ENCODING: dict[str, int] = {
    g: i for i, g in enumerate(sorted([
        'A1','A2','A3','A4','A5',
        'B1','B2','B3','B4','B5',
        'C1','C2','C3','C4','C5',
        'D1','D2','D3','D4','D5',
        'E1','E2','E3','E4','E5',
        'F1','F2','F3','F4','F5',
        'G1','G2','G3','G4','G5',
    ]))
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_sort(filepath: str) -> pd.DataFrame:
    """
    Load the LendingClub CSV and sort chronologically by issue date.

    Parameters
    ----------
    filepath : str
        Path to lc_loan.csv (e.g., 'data/raw/lc_loan.csv')

    Returns
    -------
    pd.DataFrame
        Sorted loan DataFrame.
    """
    df = pd.read_csv(filepath)
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df = df.sort_values('issue_d').reset_index(drop=True)
    return df


# ── Missing data handling ──────────────────────────────────────────────────────

def handling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - mths_since_last_delinq: fill with 300 (sentinel for 'no known delinquency')
    - emp_length: fill with mode, then extract numeric value

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # 300 = sentinel value indicating no known/recent delinquency history
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(300)

    # Mode imputation for categorical employment length
    mode_val = df['emp_length'].mode()[0]
    df['emp_length'] = df['emp_length'].fillna(mode_val)

    # Extract numeric years from strings like "10+ years" -> 10
    df['emp_length'] = df['emp_length'].apply(
        lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notnull(x) else np.nan
    )

    return df


# ── Reference table fitting ────────────────────────────────────────────────────

def fit_reference_tables(df: pd.DataFrame) -> dict:
    """
    Compute training-time reference values for dataset-relative features.
    Call this once on the training split and persist the result alongside
    the model. Pass the saved dict to create_features() at prediction time.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (after handling_data and basic feature creation).

    Returns
    -------
    dict with keys:
        'mean_int_rate_by_region'  : pd.Series  (index = zip_region)
        'mean_loan_amnt_by_purpose': pd.Series  (index = purpose)
    """
    return {
        'mean_int_rate_by_region':   df.groupby('zip_region')['int_rate'].mean(),
        'mean_loan_amnt_by_purpose': df.groupby('purpose')['loan_amnt'].mean(),
    }


# ── Feature engineering ────────────────────────────────────────────────────────

def create_features(df: pd.DataFrame, ref: dict | None = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Basic derived features (fico_avg, zip_region, stable encodings)
    2. Log transformations
    3. Interaction and squared terms
    4. Ratio and relative features
    5. Regional and purpose aggregates (uses training-fitted ref if supplied)

    Parameters
    ----------
    df : pd.DataFrame
    ref : dict or None
        Training-fitted reference tables from fit_reference_tables().
        If None, aggregates are computed from df itself (training-time only).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # ── 1. Basic derived features ──────────────────────────────────────────────
    df['fico_avg'] = (df['fico_range_high'] + df['fico_range_low']) / 2
    df['zip_region'] = df['zip_code'].astype(str).str[:3]

    # Fixed mappings — stable across all datasets; unknown category → -1
    df['purpose_encoded']   = df['purpose'].map(PURPOSE_ENCODING).fillna(-1).astype(int)
    df['sub_grade_encoded'] = df['sub_grade'].map(SUB_GRADE_ENCODING).fillna(-1).astype(int)

    # ── 2. Log transformations (reduce skewness) ───────────────────────────────
    df['log_loan_amnt']  = np.log1p(df['loan_amnt'])
    df['log_dti']        = np.log1p(df['dti'])
    df['log_revol_bal']  = np.log1p(df['revol_bal'])
    df['log_annual_inc'] = np.log1p(df['annual_inc'])

    # ── 3. Interaction terms ───────────────────────────────────────────────────
    df['grade_interest_interaction']  = df['sub_grade_encoded'] * df['int_rate']
    df['fico_int_rate_interaction']   = df['fico_avg'] * df['int_rate']
    df['fico_loan_interaction']       = df['fico_avg'] * df['loan_amnt']
    df['delinq_int_rate_interaction'] = df['mths_since_last_delinq'] * df['int_rate']
    df['dti_income_interaction']      = df['log_dti'] * df['annual_inc']
    df['grade_dti_interaction']       = df['sub_grade_encoded'] * df['log_dti']
    df['revol_loan_interaction']      = df['log_revol_bal'] * df['log_loan_amnt']
    df['purpose_loan_amnt_interaction'] = df['purpose_encoded'] * df['loan_amnt']
    df['installment_fico_interaction']  = df['installment'] * df['fico_avg']

    # ── 4. Squared terms (non-linear effects) ──────────────────────────────────
    df['int_rate_squared'] = df['int_rate'] ** 2
    df['delinq_squared']   = df['mths_since_last_delinq'] ** 2

    # ── 5. Ratios ──────────────────────────────────────────────────────────────
    df['loan_amnt_to_income_ratio']    = df['loan_amnt'] / df['annual_inc']
    df['payment_to_income_ratio']      = df['installment'] / df['annual_inc']
    df['revol_util_income_interaction'] = df['revol_util'] * df['annual_inc']
    df['fico_int_rate_ratio']          = df['fico_avg'] / df['int_rate']

    # ── 6. Dataset-relative features (anchored to training ref if supplied) ────
    if ref is not None:
        region_rates  = df['zip_region'].map(ref['mean_int_rate_by_region'])
        purpose_amnt  = df['purpose'].map(ref['mean_loan_amnt_by_purpose'])
    else:
        # Training time: compute from df itself
        region_rates = df.groupby('zip_region')['int_rate'].transform('mean')
        purpose_amnt = df.groupby('purpose')['loan_amnt'].transform('mean')

    # Fill any unseen regions/purposes with the overall mean
    region_rates = region_rates.fillna(df['int_rate'].mean())
    purpose_amnt = purpose_amnt.fillna(df['loan_amnt'].mean())

    df['mean_int_rate_by_region'] = region_rates
    df['relative_int_rate']       = df['int_rate'] / region_rates
    df['loan_to_purpose_amnt_ratio'] = df['loan_amnt'] / purpose_amnt
    df['composite_loan_feature']  = df['purpose_loan_amnt_interaction'] * df['relative_int_rate']

    return df


# ── Full preprocessing pipeline ────────────────────────────────────────────────

def run_full_pipeline(filepath: str, remove_negative_dti: bool = True) -> pd.DataFrame:
    """
    Convenience function: load -> handle missing -> remove bad rows -> feature engineer.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.
    remove_negative_dti : bool
        Whether to drop rows with dti < 0. Default True.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame ready for modelling.
    """
    df = load_and_sort(filepath)
    df = handling_data(df)
    if remove_negative_dti:
        df = df[df['dti'] >= 0].reset_index(drop=True)
    df = create_features(df)
    return df
