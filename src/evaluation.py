"""
evaluation.py
-------------
Model validation utilities for the Credit Risk Modeling project.

Implements credit-risk-specific metrics used by model validators:
  - KS Statistic (Kolmogorov-Smirnov): standard credit scorecard discrimination metric
  - Population Stability Index (PSI): detects score distribution drift between windows
  - Threshold Analysis Table: business cost at different decision thresholds
  - SHAP wrapper: unified SHAP analysis for tree and linear models

Usage:
    from src.evaluation import ks_statistic, psi, threshold_analysis_table, shap_analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score,
)


# ── KS Statistic ───────────────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic for binary classification.

    The KS stat is the maximum difference between the cumulative distribution
    functions of predicted scores for defaulters vs non-defaulters. Banks use
    this as the primary discrimination metric for credit scorecards — not ROC-AUC alone.

    A KS > 0.20 is generally considered acceptable for credit scoring;
    > 0.40 is strong.

    Parameters
    ----------
    y_true : array-like
        Binary labels (1 = default, 0 = non-default).
    y_prob : array-like
        Predicted probabilities for the positive class (default).

    Returns
    -------
    dict with keys:
        'ks_stat' : float — the KS statistic
        'ks_pvalue' : float — p-value from the two-sample KS test
        'interpretation' : str — qualitative rating
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    scores_default = y_prob[y_true == 1]
    scores_non_default = y_prob[y_true == 0]

    ks_stat, ks_pvalue = stats.ks_2samp(scores_default, scores_non_default)

    if ks_stat >= 0.40:
        interpretation = "Strong (KS ≥ 0.40)"
    elif ks_stat >= 0.20:
        interpretation = "Acceptable (0.20 ≤ KS < 0.40)"
    else:
        interpretation = "Weak (KS < 0.20)"

    return {
        'ks_stat': round(ks_stat, 4),
        'ks_pvalue': round(ks_pvalue, 6),
        'interpretation': interpretation,
    }


def plot_ks_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model"):
    """
    Plot the KS separation chart: cumulative default rate vs non-default rate.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    n = len(df)
    df['cum_default'] = df['y_true'].cumsum() / df['y_true'].sum()
    df['cum_non_default'] = (1 - df['y_true']).cumsum() / (1 - df['y_true']).sum()
    df['ks_diff'] = df['cum_default'] - df['cum_non_default']

    ks_idx = df['ks_diff'].abs().idxmax()
    ks_val = df.loc[ks_idx, 'ks_diff']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.linspace(0, 1, n), df['cum_default'], label='Cumulative Default Rate', color='red')
    ax.plot(np.linspace(0, 1, n), df['cum_non_default'], label='Cumulative Non-Default Rate', color='blue')
    ax.axvline(x=ks_idx / n, linestyle='--', color='gray', alpha=0.7)
    ax.annotate(f'KS = {abs(ks_val):.3f}', xy=(ks_idx / n, (df.loc[ks_idx, 'cum_default'] + df.loc[ks_idx, 'cum_non_default']) / 2),
                fontsize=12, color='black')
    ax.set_title(f'KS Separation Chart — {model_name}')
    ax.set_xlabel('Proportion of Population (sorted by score, descending)')
    ax.set_ylabel('Cumulative Rate')
    ax.legend()
    plt.tight_layout()
    return fig


# ── Population Stability Index (PSI) ──────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute the Population Stability Index (PSI) between two score distributions.

    PSI measures how much a model's score distribution has shifted between
    a reference period (e.g., training window) and a monitoring period (e.g.,
    test window). Model validators use PSI to detect model drift before
    performance degradation becomes visible in AUC.

    PSI interpretation (industry standard):
      PSI < 0.10  : No significant shift — model is stable
      0.10 ≤ PSI < 0.25 : Moderate shift — investigate
      PSI ≥ 0.25  : Large shift — model should be rebuilt

    Parameters
    ----------
    expected : array-like
        Score distribution from the reference/training period.
    actual : array-like
        Score distribution from the monitoring/test period.
    n_bins : int
        Number of bins for the score distribution. Default 10.

    Returns
    -------
    dict with keys:
        'psi' : float — the PSI value
        'interpretation' : str — qualitative rating
        'bin_psi' : pd.DataFrame — per-bin PSI breakdown
    """
    eps = 1e-4  # avoid log(0)
    breakpoints = np.linspace(0, 1, n_bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Clip zeros to avoid log(0)
    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)

    bin_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = bin_psi.sum()

    if total_psi < 0.10:
        interpretation = "Stable (PSI < 0.10)"
    elif total_psi < 0.25:
        interpretation = "Moderate shift — investigate (0.10 ≤ PSI < 0.25)"
    else:
        interpretation = "Large shift — model rebuild required (PSI ≥ 0.25)"

    bin_df = pd.DataFrame({
        'Bin': [f'{breakpoints[i]:.1f}–{breakpoints[i+1]:.1f}' for i in range(n_bins)],
        'Expected %': (expected_pct * 100).round(2),
        'Actual %': (actual_pct * 100).round(2),
        'Bin PSI': bin_psi.round(4),
    })

    return {
        'psi': round(total_psi, 4),
        'interpretation': interpretation,
        'bin_psi': bin_df,
    }


# ── Threshold Analysis Table ───────────────────────────────────────────────────

def threshold_analysis_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list = None,
    cost_fn: float = 1.0,
    cost_fp: float = 5.0,
) -> pd.DataFrame:
    """
    Business-oriented threshold analysis.

    For each decision threshold, compute classification metrics AND an estimated
    business cost based on rejected good loans (FP) vs approved bad loans (FN).
    This reframes the model from 'academic classifier' to 'credit decision tool'.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_prob : array-like
        Predicted probabilities.
    thresholds : list of float
        Decision thresholds to evaluate. Default [0.1, 0.2, 0.3, 0.4, 0.5].
    cost_fn : float
        Relative cost of a false negative (approved bad loan). Default 1.0.
    cost_fp : float
        Relative cost of a false positive (rejected good loan). Default 5.0.
        Interpretation: rejecting a good loan costs 5× a bad loan slipping through.

    Returns
    -------
    pd.DataFrame
        One row per threshold with: Threshold, Accuracy, Precision, Recall,
        F1, ROC-AUC, #FP, #FN, Business Cost.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append({
            'Threshold': t,
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'ROC-AUC': round(auc, 4),
            'False Positives (Good loans rejected)': fp,
            'False Negatives (Bad loans approved)': fn,
            f'Business Cost (FP×{cost_fp} + FN×{cost_fn})': round(fp * cost_fp + fn * cost_fn, 0),
        })

    return pd.DataFrame(rows)


# ── Classification metrics utility ────────────────────────────────────────────

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all key classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


# ── SHAP analysis wrapper ──────────────────────────────────────────────────────

def shap_analysis(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    palette: str = 'Blues_d',
    is_tree_based: bool = True,
    top_n: int = 10,
):
    """
    Compute SHAP values and produce a bar chart of top-N most important features.

    Parameters
    ----------
    model : fitted model object
    X_train : np.ndarray — training data (dense)
    X_test : np.ndarray — test data for SHAP computation (dense)
    feature_names : list of str
    model_name : str — used in plot title
    palette : str — seaborn colour palette
    is_tree_based : bool — True for XGBoost/LightGBM/RF, False for Ridge/Lasso
    top_n : int — number of top features to plot

    Returns
    -------
    tuple : (shap_values, top_features_df)
    """
    if is_tree_based:
        explainer = shap.TreeExplainer(model, X_train)
    else:
        explainer = shap.LinearExplainer(model, X_train)

    shap_values = explainer.shap_values(X_test)

    # Handle binary classification output (use class 1 values)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_shap,
    }).sort_values('Mean |SHAP|', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Mean |SHAP|', y='Feature', data=top_df, palette=palette, ax=ax)
    ax.set_title(f'Top {top_n} Features by Mean |SHAP| — {model_name}')
    plt.tight_layout()
    plt.show()

    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=True)

    return shap_values, top_df
