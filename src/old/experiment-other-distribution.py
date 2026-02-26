
# -*- coding: utf-8 -*-
"""
Single-plot overlay: Babbling vs Maximally Informative across priors
Priors: Uniform, Pareto, Exponential (all normalized to [0,1]).

- Utility: threshold (eps_order=0)
- X-axis: constant bias level b ∈ [0,1]
- Y-axis: expected user utility U(q) = sum_v theta[v] * a[v]
- One figure with all curves:
    For each prior: solid = Max-Info, dashed = Babbling
- No markers; clean colors and grid.

Requires: coi_algorithms.py available in import path.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import lgamma, exp

if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    compute_expected_posteriors, system_best_response,
    algorithm_2_build_qbase, algorithm_4_maximally_informative,
    user_utility_from_response
)

# ----------------------------
# Data loading
# ----------------------------

CANDIDATE_HOURS_COLS = [
    'hours_per_week', 'hours-per-week', 'hoursPerWeek', 'hours', 'hours.week', 'hours-per.week'
]

def load_census_hours_csv(dataset_path: str) -> pd.Series:
    df = pd.read_csv(dataset_path)
    col = None
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in CANDIDATE_HOURS_COLS:
        if cand in df.columns:
            col = cand; break
        if cand.lower() in lower_cols:
            col = lower_cols[cand.lower()]; break
    if col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No numeric column found; please specify the hours-per-week column.")
        uniques = sorted([(c, df[c].nunique(dropna=True)) for c in num_cols], key=lambda x: x[1])
        col = uniques[len(uniques)//2][0]
        print(f"[WARN] Falling back to numeric column '{col}'. Override if incorrect.")
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    return s

# ----------------------------
# Prior profiles (normalized [0,1])
# ----------------------------

def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr)); hi = float(np.max(arr))
    if hi == lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)

def uniform_rank_profile(k: int) -> List[float]:
    exps_asc = np.array([j/(k+1.0) for j in range(1, k+1)], dtype=float)
    return exps_asc[::-1].tolist()

def pareto_rank_profile_normalized(k: int, alpha: float = 1.5, x_min: float = 1.0) -> List[float]:
    if alpha <= 1.0:
        raise ValueError("Pareto expectation requires alpha > 1.")
    exps_asc = []
    for j in range(1, k+1):
        a1 = j - 1.0/alpha
        b1 = k - j + 1
        if a1 <= 0:
            exps_asc.append(float('inf'))
        else:
            log_num = _log_beta(a1, b1)
            log_den = _log_beta(j, b1)
            exps_asc.append(x_min * exp(log_num - log_den))
    phi_desc = np.array(exps_asc[::-1], dtype=float)
    return _minmax_norm(phi_desc).tolist()

def exp_rank_profile_normalized(k: int, lam: float = 1.0) -> List[float]:
    H = np.zeros(k+1, dtype=float)
    for n in range(1, k+1):
        H[n] = H[n-1] + 1.0/n
    exps_asc = (H[k] - H[:k]) / float(lam)
    phi_desc = np.array(exps_asc[::-1], dtype=float)
    return _minmax_norm(phi_desc).tolist()

def make_prior_spec_for_domain(k: int, kind: str, **kwargs) -> PriorSpec:
    if kind == 'uniform':
        phi = uniform_rank_profile(k)
    elif kind == 'pareto':
        phi = pareto_rank_profile_normalized(k, alpha=float(kwargs.get('alpha', 1.5)), x_min=1.0)
    elif kind == 'exponential':
        phi = exp_rank_profile_normalized(k, lam=float(kwargs.get('lam', 1.0)))
    else:
        raise ValueError("Unknown prior kind")
    def custom_rank_expectation(k_arg: int, r_desc: int) -> float:
        if k_arg != k:
            idx = max(1, min(r_desc, k)) - 1
            return float(phi[idx])
        return float(phi[r_desc - 1])
    return PriorSpec(kind='custom', custom=custom_rank_expectation)

# ----------------------------
# Bias helper
# ----------------------------

def make_constant_bias(level: float) -> Bias1D:
    return Bias1D(kind='constant', base=float(level))

# ----------------------------
# Utility
# ----------------------------

def expected_utility_for_query(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec, biases: Dict[float, float]
) -> float:
    theta_singleton = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta = system_best_response(q_groups, domain, prior, biases, receiver_model='threshold')
    return float(user_utility_from_response(theta_singleton, beta, receiver_model='threshold', order_list=domain, eps_order=0.0))

# ----------------------------
# Runner (single combined plot)
# ----------------------------

def run_multi_prior_single_plot(
    dataset_path: str,
    bias_grid: List[float],
    prior_specs: List[Tuple[str, Dict]],
    output_prefix: Optional[str] = None
) -> pd.DataFrame:
    s = load_census_hours_csv(dataset_path)
    domain_values = s.dropna().unique().tolist()
    domain = dedupe_and_sort_desc(domain_values)
    k = len(domain)

    # Prepare plot
    plt.figure()
    color_map = {
        'uniform': '#1f77b4',      # blue
        'pareto': '#d62728',       # red
        'exponential': '#2ca02c',  # green
    }

    frames = []
    for kind, params in prior_specs:
        prior = make_prior_spec_for_domain(k, kind, **params)
        ys_babble, ys_star = [], []
        for b in bias_grid:
            biases = biases_from_bias_obj(domain, make_constant_bias(b))
            # Babbling
            u_babble = expected_utility_for_query([domain[:]], domain, prior, biases)
            # Max-Info
            q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, receiver_model='threshold')
            q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, receiver_model='threshold')
            u_star = expected_utility_for_query(q_star, domain, prior, biases)
            ys_babble.append(u_babble); ys_star.append(u_star)

        # Plot lines (no markers)
        c = color_map.get(kind, None)
        # nomalize utility to [0,1] for better comparison
        # ys_babble = np.array(ys_babble) / max(ys_babble) if max(ys_babble) > 0 else ys_babble
        # ys_star = np.array(ys_star) / max(ys_star) if max(ys_star) > 0 else ys_star
        def _norm(y):
            y_min, y_max = float(np.min(y)), float(np.max(y))
            return (y - y_min) / (y_max - y_min) if y_max > y_min else y * 0.0

        ys_babble = _norm(np.array(ys_babble))
        ys_star = _norm(np.array(ys_star))


        plt.plot(bias_grid, ys_star, linewidth=2.8, linestyle='-',  color=c, label=f'{kind.capitalize()} — Max-Info')
        plt.plot(bias_grid, ys_babble, linewidth=2.2, linestyle='--', color=c, label=f'{kind.capitalize()} — UnInfo')


        df_prior = pd.DataFrame({'prior': kind, 'bias_level': bias_grid, 'utility_babbling': ys_babble, 'utility_maxinfo': ys_star})
        frames.append(df_prior)

    plt.xlabel('Bias Degree')
    plt.ylabel('Expected User Utility')
    plt.title('User Utility on Uninformative vs Maximally Informative Equilibria')
    plt.legend(frameon=False, ncols=1)
    #move legend to top right and make it smaller move further right
    plt.legend(loc='upper right', fontsize='small', frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_df = pd.concat(frames, ignore_index=True)

    if output_prefix:
        out_dir = Path(output_prefix).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = f"{output_prefix}multi_prior_results.csv"
        png_path = f"{output_prefix}multi_prior_combined_plot.png"
        out_df.to_csv(csv_path, index=False)
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Wrote CSV to: {csv_path}")
        print(f"Wrote PNG to: {png_path}")
    else:
        plt.close()

    return out_df

# ----------------------------
# Main
# ----------------------------

if __name__ == '__main__':
    DATASET_PATH = '../data/real/census.csv'
    OUTPUT_PREFIX = '/Users/aryal/Desktop/Querying-COI/results/babble_vs_maxinfo_multi_prior_combined_'
    bias_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Choose parameters to ensure top ranks have higher theta after normalization
    prior_specs = [
        ('uniform', {}),
        ('pareto', {'alpha': 2}),     # top-heavy
        ('exponential', {'lam': 1.0}),  # reasonable decay
    ]

    try:
        _ = run_multi_prior_single_plot(DATASET_PATH, bias_levels, prior_specs, output_prefix=OUTPUT_PREFIX)
    except FileNotFoundError:
        print(f"Dataset not found at '{DATASET_PATH}'. Please set DATASET_PATH to your census CSV.")

