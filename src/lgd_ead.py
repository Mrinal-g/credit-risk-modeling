"""
lgd_ead.py
----------
LGD (Loss Given Default), EAD (Exposure at Default), and Expected Loss framework.
Also implements IFRS 9 Stage classification and stress testing sensitivity analysis.

This module completes the full PD / LGD / EAD credit risk framework:

    Expected Loss = PD × LGD × EAD

PD  — predicted by XGBoost classifier (notebook 03)
LGD — predicted by Ridge/LightGBM regression on recovery rate (this module)
EAD — computed from outstanding principal (this module)

IFRS 9 / CECL context:
    The expected loss framework implemented here is consistent with the lifetime
    expected credit loss (ECL) approach required under IFRS 9 and CECL. Stage 1/2/3
    bucket classification is implemented below.

Usage:
    from src.lgd_ead import (
        prepare_lgd_data, train_lgd_model, compute_ead,
        compute_expected_loss, ifrs9_stage_classify,
        stress_test_portfolio
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
# LGD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def prepare_lgd_data(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the LGD training dataset.

    LGD = 1 - recovery_rate
    recovery_rate = recoveries / funded_amnt  (clipped to [0, 1])

    Only loans that actually defaulted are used to train the LGD model —
    recovery data only exists for defaulted loans.

    Parameters
    ----------
    loan_data : pd.DataFrame
        Full loan dataset after feature engineering (from preprocessing.py).

    Returns
    -------
    pd.DataFrame
        Defaulted loans only, with new columns:
        - recovery_rate : float in [0, 1]
        - lgd           : float in [0, 1]  (target for regression)
    """
    defaulted = loan_data[loan_data['early_default'] == 1].copy()

    if 'recoveries' in defaulted.columns and 'funded_amnt' in defaulted.columns:
        # Full LendingClub dataset: use actual recovery amounts
        defaulted['recovery_rate'] = (
            defaulted['recoveries'] / defaulted['funded_amnt'].replace(0, np.nan)
        ).clip(0, 1).fillna(0)
    elif 'return' in defaulted.columns:
        # Preprocessed dataset: derive recovery from loan return
        # return = (total_received / funded_amnt) - 1, so recovery_rate = 1 + return
        defaulted['recovery_rate'] = (1 + defaulted['return']).clip(0, 1)
    else:
        raise ValueError(
            "Cannot compute LGD: dataset must contain either 'recoveries'+'funded_amnt' "
            "or a 'return' column."
        )

    defaulted['lgd'] = 1 - defaulted['recovery_rate']

    print(f"Defaulted loans for LGD training: {len(defaulted):,}")
    print(f"Mean recovery rate:  {defaulted['recovery_rate'].mean():.2%}")
    print(f"Mean LGD:            {defaulted['lgd'].mean():.2%}")
    print(f"LGD distribution:\n{defaulted['lgd'].describe().round(4)}")

    return defaulted


def train_lgd_model(
    defaulted_df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a Ridge regression LGD model.

    LGD is bounded in [0, 1], so we use a Ridge regression and clip
    predictions to that range. GradientBoosting is offered as an alternative
    for comparison — it often captures the bimodal distribution of LGD better.

    Parameters
    ----------
    defaulted_df : pd.DataFrame
        Output of prepare_lgd_data().
    feature_cols : list of str
        Feature columns to use (numerical only — no categoricals for speed).
    test_size : float
        Fraction of defaulted loans held out for evaluation.
    random_state : int

    Returns
    -------
    dict with keys:
        'model'         : fitted Ridge model
        'r2_in'         : in-sample R²
        'r2_out'        : out-of-sample R²
        'mae_out'       : out-of-sample MAE
        'y_test'        : true LGD values on test set
        'y_pred'        : predicted LGD values on test set
        'scaler'        : fitted StandardScaler
    """
    from sklearn.model_selection import train_test_split

    # Keep only rows where feature_cols are all present
    df = defaulted_df[feature_cols + ['lgd']].dropna()
    X = df[feature_cols].values
    y = df['lgd'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Ridge with cross-validated alpha
    ridge_rs = RandomizedSearchCV(
        Ridge(), {'alpha': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]},
        n_iter=8, scoring='neg_mean_squared_error', cv=5,
        random_state=random_state, n_jobs=-1
    )
    ridge_rs.fit(X_train_s, y_train)
    model = ridge_rs.best_estimator_

    y_pred_train = np.clip(model.predict(X_train_s), 0, 1)
    y_pred_test  = np.clip(model.predict(X_test_s), 0, 1)

    results = {
        'model':   model,
        'scaler':  scaler,
        'r2_in':   r2_score(y_train, y_pred_train),
        'r2_out':  r2_score(y_test,  y_pred_test),
        'mae_out': mean_absolute_error(y_test, y_pred_test),
        'y_test':  y_test,
        'y_pred':  y_pred_test,
        'best_alpha': ridge_rs.best_params_['alpha'],
        'feature_cols': feature_cols,
    }

    print(f"\nLGD Model — Ridge (alpha={results['best_alpha']})")
    print(f"  In-sample  R²  : {results['r2_in']:.4f}")
    print(f"  Out-of-sample R²  : {results['r2_out']:.4f}")
    print(f"  Out-of-sample MAE : {results['mae_out']:.4f}")
    print(f"\n  Interpretation: MAE of {results['mae_out']:.2%} means the model's")
    print(f"  LGD predictions are off by ~{results['mae_out']:.2%} on average.")

    return results


def plot_lgd_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot true vs predicted LGD distribution.
    LGD for P2P loans is bimodal: near 0 (full recovery) or near 1 (no recovery).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_true, bins=30, color='steelblue', alpha=0.7, label='True LGD')
    axes[0].hist(y_pred, bins=30, color='coral', alpha=0.6, label='Predicted LGD')
    axes[0].set_title('True vs Predicted LGD Distribution')
    axes[0].set_xlabel('LGD')
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, alpha=0.2, s=5, color='steelblue')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect prediction')
    axes[1].set_xlabel('True LGD')
    axes[1].set_ylabel('Predicted LGD')
    axes[1].set_title('Predicted vs Actual LGD')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# EAD CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ead(loan_data: pd.DataFrame) -> pd.Series:
    """
    Compute Exposure at Default (EAD) for each loan.

    For term loans (fixed schedule), EAD = outstanding principal at time of
    default. For performing loans, EAD = current outstanding principal.

    LendingClub provides:
    - out_prncp       : outstanding principal (for performing loans)
    - out_prncp_inv   : investor-held outstanding principal
    - loan_amnt       : original loan amount (fallback if out_prncp is 0)

    The Credit Conversion Factor (CCF) is 1.0 for term loans (fully drawn).

    Parameters
    ----------
    loan_data : pd.DataFrame

    Returns
    -------
    pd.Series
        EAD per loan (in original currency units, typically USD).
    """
    if 'out_prncp' in loan_data.columns:
        ead = loan_data['out_prncp'].copy()
        # For loans where out_prncp is 0 or missing, fall back to funded_amnt
        fallback_mask = (ead <= 0) | ead.isna()
        if 'funded_amnt' in loan_data.columns:
            ead[fallback_mask] = loan_data.loc[fallback_mask, 'funded_amnt']
        else:
            ead[fallback_mask] = loan_data.loc[fallback_mask, 'loan_amnt']
    elif 'funded_amnt' in loan_data.columns:
        ead = loan_data['funded_amnt'].copy()
    else:
        ead = loan_data['loan_amnt'].copy()

    print(f"EAD statistics:")
    print(f"  Mean EAD: ${ead.mean():,.0f}")
    print(f"  Median EAD: ${ead.median():,.0f}")
    print(f"  Total portfolio EAD: ${ead.sum():,.0f}")

    return ead


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED LOSS = PD × LGD × EAD
# ══════════════════════════════════════════════════════════════════════════════

def compute_expected_loss(
    pd_scores: np.ndarray,
    lgd_scores: np.ndarray,
    ead_values: np.ndarray,
    loan_ids: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute Expected Loss (EL) at loan level and portfolio level.

    Formula: EL = PD × LGD × EAD

    Parameters
    ----------
    pd_scores  : array-like, float in [0, 1]
        Predicted probability of default from XGBoost PD model.
    lgd_scores : array-like, float in [0, 1]
        Predicted LGD from Ridge LGD model.
    ead_values : array-like, float > 0
        EAD per loan (USD).
    loan_ids   : pd.Series, optional
        Loan identifiers for the output DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        loan_id (if provided), pd, lgd, ead, expected_loss, el_rate
    """
    pd_arr  = np.asarray(pd_scores,  dtype=float)
    lgd_arr = np.asarray(lgd_scores, dtype=float)
    ead_arr = np.asarray(ead_values, dtype=float)

    el = pd_arr * lgd_arr * ead_arr
    el_rate = pd_arr * lgd_arr  # EL as % of exposure

    df = pd.DataFrame({
        'pd':             pd_arr.round(6),
        'lgd':            lgd_arr.round(6),
        'ead':            ead_arr.round(2),
        'expected_loss':  el.round(2),
        'el_rate':        el_rate.round(6),
    })

    if loan_ids is not None:
        df.insert(0, 'loan_id', loan_ids.values)

    print(f"\nPortfolio Expected Loss Summary:")
    print(f"  Total EAD:             ${ead_arr.sum():>15,.0f}")
    print(f"  Total Expected Loss:   ${el.sum():>15,.0f}")
    print(f"  Portfolio EL Rate:     {el.sum() / ead_arr.sum():.2%}")
    print(f"  Mean loan-level PD:    {pd_arr.mean():.2%}")
    print(f"  Mean loan-level LGD:   {lgd_arr.mean():.2%}")
    print(f"  Mean loan-level EAD:   ${ead_arr.mean():,.0f}")

    return df


def plot_expected_loss_breakdown(el_df: pd.DataFrame):
    """Portfolio EL visualizations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(el_df['pd'], bins=40, color='coral', alpha=0.8)
    axes[0].set_title('PD distribution')
    axes[0].set_xlabel('Probability of Default')
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[1].hist(el_df['lgd'], bins=40, color='steelblue', alpha=0.8)
    axes[1].set_title('LGD distribution')
    axes[1].set_xlabel('Loss Given Default')
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[2].hist(el_df['expected_loss'], bins=40, color='purple', alpha=0.8)
    axes[2].set_title('Expected Loss distribution')
    axes[2].set_xlabel('Expected Loss ($)')
    axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    plt.suptitle('Portfolio Expected Loss = PD × LGD × EAD', y=1.02)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# IFRS 9 STAGE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def ifrs9_stage_classify(
    pd_scores: np.ndarray,
    stage2_threshold: float = 0.10,
    stage3_threshold: float = 0.30,
) -> np.ndarray:
    """
    Classify loans into IFRS 9 / CECL stages based on PD.

    IFRS 9 requires banks to estimate Expected Credit Loss (ECL) over:
    - Stage 1: 12-month ECL  (low risk — PD has not significantly increased)
    - Stage 2: Lifetime ECL  (significant increase in credit risk since origination)
    - Stage 3: Lifetime ECL  (credit-impaired — objective evidence of default)

    Thresholds used here are illustrative; real banks use origination PD
    vs current PD comparison (PD deterioration triggers Stage 2 migration).

    Parameters
    ----------
    pd_scores        : array-like, float in [0, 1]
    stage2_threshold : float — PD above this → Stage 2 (default 10%)
    stage3_threshold : float — PD above this → Stage 3 (default 30%)

    Returns
    -------
    np.ndarray of int (1, 2, or 3)
    """
    pd_arr = np.asarray(pd_scores, dtype=float)
    stages = np.ones(len(pd_arr), dtype=int)  # default Stage 1
    stages[pd_arr >= stage2_threshold] = 2
    stages[pd_arr >= stage3_threshold] = 3
    return stages


def ifrs9_summary(pd_scores: np.ndarray, el_df: pd.DataFrame,
                  stage2_threshold: float = 0.10,
                  stage3_threshold: float = 0.30) -> pd.DataFrame:
    """
    Produce an IFRS 9 staging summary table.

    Shows loan count, EAD, and Expected Loss by Stage.
    """
    stages = ifrs9_stage_classify(pd_scores, stage2_threshold, stage3_threshold)
    el_df = el_df.copy()
    el_df['stage'] = stages

    summary = el_df.groupby('stage').agg(
        loan_count=('pd', 'count'),
        total_ead=('ead', 'sum'),
        total_el=('expected_loss', 'sum'),
        mean_pd=('pd', 'mean'),
        mean_lgd=('lgd', 'mean'),
    ).round(4)

    summary['ead_pct']  = (summary['total_ead'] / summary['total_ead'].sum() * 100).round(2)
    summary['el_rate']  = (summary['total_el']  / summary['total_ead'] * 100).round(4)

    print("\nIFRS 9 / CECL Stage Classification Summary")
    print("=" * 65)
    print(f"Stage 1 (12-month ECL):   PD < {stage2_threshold:.0%}")
    print(f"Stage 2 (Lifetime ECL):   {stage2_threshold:.0%} ≤ PD < {stage3_threshold:.0%}")
    print(f"Stage 3 (Credit-impaired): PD ≥ {stage3_threshold:.0%}")
    print()
    print(summary.to_string())
    print()
    print("Note: In practice, Stage migration is triggered by PD deterioration")
    print("vs origination PD, not absolute PD thresholds alone.")

    return summary


def plot_ifrs9_stages(el_df: pd.DataFrame, pd_scores: np.ndarray,
                      stage2_threshold: float = 0.10,
                      stage3_threshold: float = 0.30):
    """Bar charts of loans and EAD by IFRS 9 stage."""
    el_df = el_df.copy()
    el_df['stage'] = ifrs9_stage_classify(pd_scores, stage2_threshold, stage3_threshold)

    stage_counts = el_df.groupby('stage').agg(
        loan_count=('pd', 'count'),
        total_ead=('ead', 'sum'),
        total_el=('expected_loss', 'sum'),
    )

    colors = {1: 'steelblue', 2: 'orange', 3: 'coral'}
    stage_labels = {1: 'Stage 1\n(12-month ECL)', 2: 'Stage 2\n(Lifetime ECL)', 3: 'Stage 3\n(Credit-impaired)'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, (col, title, fmt) in enumerate([
        ('loan_count', 'Loan count by stage', '{:,.0f}'),
        ('total_ead',  'Total EAD by stage ($)', '${:,.0f}'),
        ('total_el',   'Expected Loss by stage ($)', '${:,.0f}'),
    ]):
        bar_colors = [colors[s] for s in stage_counts.index]
        bars = axes[i].bar(
            [stage_labels[s] for s in stage_counts.index],
            stage_counts[col],
            color=bar_colors, edgecolor='none', alpha=0.85
        )
        axes[i].set_title(title)
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.01,
                         fmt.format(bar.get_height()),
                         ha='center', va='bottom', fontsize=8)

    plt.suptitle('IFRS 9 Stage Distribution', y=1.02)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════

STRESS_SCENARIOS = {
    'Base (current)':        {'pd_multiplier': 1.00, 'lgd_multiplier': 1.00},
    'Mild stress':           {'pd_multiplier': 1.25, 'lgd_multiplier': 1.05},
    'Moderate stress':       {'pd_multiplier': 1.50, 'lgd_multiplier': 1.10},
    'Severe stress':         {'pd_multiplier': 2.00, 'lgd_multiplier': 1.20},
    'Extreme (crisis)':      {'pd_multiplier': 3.00, 'lgd_multiplier': 1.30},
}


def stress_test_portfolio(
    pd_scores: np.ndarray,
    lgd_scores: np.ndarray,
    ead_values: np.ndarray,
    scenarios: dict = None,
) -> pd.DataFrame:
    """
    Run portfolio stress tests across multiple macro scenarios.

    Each scenario applies a PD multiplier and LGD multiplier to simulate
    adverse macro conditions (recession, credit crisis). This is conceptually
    consistent with DFAST / CCAR scenario analysis, but uses simple scalar
    shocks rather than modeled macro sensitivities.

    Note: PD is clipped to [0, 1] after scaling. LGD is clipped to [0, 1].

    Parameters
    ----------
    pd_scores   : array-like — base PD predictions
    lgd_scores  : array-like — base LGD predictions
    ead_values  : array-like — EAD per loan
    scenarios   : dict, optional
        Override default scenarios. Format:
        {'Scenario name': {'pd_multiplier': float, 'lgd_multiplier': float}}

    Returns
    -------
    pd.DataFrame
        One row per scenario with: total EAD, total EL, EL rate, EL change vs base.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    pd_arr  = np.asarray(pd_scores,  dtype=float)
    lgd_arr = np.asarray(lgd_scores, dtype=float)
    ead_arr = np.asarray(ead_values, dtype=float)
    total_ead = ead_arr.sum()

    rows = []
    base_el = None

    for name, params in scenarios.items():
        stressed_pd  = np.clip(pd_arr  * params['pd_multiplier'],  0, 1)
        stressed_lgd = np.clip(lgd_arr * params['lgd_multiplier'], 0, 1)
        stressed_el  = (stressed_pd * stressed_lgd * ead_arr).sum()
        el_rate      = stressed_el / total_ead

        if base_el is None:
            base_el = stressed_el

        rows.append({
            'Scenario':             name,
            'PD multiplier':        f'{params["pd_multiplier"]:.2f}×',
            'LGD multiplier':       f'{params["lgd_multiplier"]:.2f}×',
            'Portfolio EAD ($)':    f'${total_ead:,.0f}',
            'Expected Loss ($)':    f'${stressed_el:,.0f}',
            'EL Rate':              f'{el_rate:.2%}',
            'EL vs Base':           f'+{(stressed_el - base_el):,.0f} ({(stressed_el/base_el - 1):.1%})'
                                    if stressed_el > base_el else 'Base',
        })

    result_df = pd.DataFrame(rows)
    print("\nPortfolio Stress Test Results")
    print("=" * 95)
    print(result_df.to_string(index=False))
    print()
    print("Note: PD multipliers simulate macroeconomic stress (e.g., 2.0× PD = severe recession).")
    print("LGD multipliers simulate collateral value decline and reduced recovery in downturns.")
    print("This is illustrative stress sensitivity — not a DFAST/CCAR-compliant macro model.")

    return result_df


def plot_stress_test(
    pd_scores: np.ndarray,
    lgd_scores: np.ndarray,
    ead_values: np.ndarray,
    scenarios: dict = None,
):
    """Bar chart of Expected Loss across stress scenarios."""
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    pd_arr  = np.asarray(pd_scores,  dtype=float)
    lgd_arr = np.asarray(lgd_scores, dtype=float)
    ead_arr = np.asarray(ead_values, dtype=float)

    names = []
    el_values = []

    for name, params in scenarios.items():
        stressed_pd  = np.clip(pd_arr  * params['pd_multiplier'],  0, 1)
        stressed_lgd = np.clip(lgd_arr * params['lgd_multiplier'], 0, 1)
        names.append(name)
        el_values.append((stressed_pd * stressed_lgd * ead_arr).sum())

    palette = ['steelblue', 'dodgerblue', 'orange', 'coral', 'crimson']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, el_values, color=palette[:len(names)], edgecolor='none', alpha=0.85)

    for bar, val in zip(bars, el_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_title('Portfolio Expected Loss — Stress Scenarios\n(PD × LGD × EAD)', fontsize=12)
    ax.set_ylabel('Total Expected Loss ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.show()
