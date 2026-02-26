

# -*- coding: utf-8 -*-
"""
Babbling vs. Maximally Informative — Real Census (hours-per-week)

Goal
----
Compare the expected user utility under:
  (1) Babbling baseline (single group; system ignores the order)
  (2) Maximally informative pipeline (Alg.2 + Alg.4)

Setup
-----
- Utility: threshold (as in coi_algorithms.user_utility_from_response for 'threshold')
- Prior: uniform [0,1] via PriorSpec(kind="uniform") → rank-based expectations in [0,1]
- Bias: scalar level b ∈ [0,1] applied uniformly to all values (bias[v] = b)
- X-axis: bias level b
- Y-axis: expected user utility (sum over domain of θ[v] * action[v])

Notes
-----
- For babbling, the expected posterior within the single group is the group average
  (which equals ~0.5 under the uniform-rank prior), so action is 1{0.5 > b}.
- We set eps_order=0 to disable tiny order bonuses, so the comparison is purely threshold-based.
- Uses the real-world census dataset's "hours-per-week" attribute (robust column matching).
"""

import sys, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local module is importable if running from a different working dir.
if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    compute_expected_posteriors, system_best_response,
    algorithm_2_build_qbase, algorithm_4_maximally_informative,
    user_utility_from_response, flatten_in_order
)

# ----------------------------
# Data loading
# ----------------------------

CANDIDATE_HOURS_COLS = [
    'hours_per_week', 'hours-per-week', 'hoursPerWeek', 'hours', 'hours.week', 'hours-per.week'
]

def load_census_hours_csv(dataset_path: str) -> pd.Series:
    df = pd.read_csv(dataset_path)
    # Try to find the hours-per-week column (robust to different namings)
    col = None
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in CANDIDATE_HOURS_COLS:
        if cand in df.columns:
            col = cand
            break
        if cand.lower() in lower_cols:
            col = lower_cols[cand.lower()]
            break
    if col is None:
        # Last resort: choose the first numeric column with many distinct values
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No numeric column found; please specify the hours-per-week column.")
        # Heuristic: pick the numeric column with the median number of uniques (avoid IDs and tiny cardinality)
        uniques = sorted([(c, df[c].nunique(dropna=True)) for c in num_cols], key=lambda x: x[1])
        col = uniques[len(uniques)//2][0]
        print(f"[WARN] Falling back to numeric column '{col}'. Override if incorrect.")
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    return s

# ----------------------------
# Bias helpers
# ----------------------------

def make_constant_bias(level: float) -> Bias1D:
    """Uniform bias across the domain: bias[v] = level ∈ [0,1]."""
    return Bias1D(kind='constant', base=float(level))

# ----------------------------
# Utility computation
# ----------------------------

def expected_utility_for_query(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec, biases: Dict[float, float]
) -> float:
    """
    Expected user utility under the threshold receiver for a given query q_groups,
    measuring utility with respect to the user's intent theta (singleton ranking).
    eps_order = 0 so we ignore any tiny order bonus.
    """
    theta_singleton = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta = system_best_response(q_groups, domain, prior, biases, receiver_model='threshold')
    # order_list is irrelevant when eps_order=0, but we pass domain anyway
    u = user_utility_from_response(theta_singleton, beta, receiver_model='threshold', order_list=domain, eps_order=0.0)
    return float(u)

# ----------------------------
# Experiment runner
# ----------------------------

def run_babble_vs_maxinfo(
    dataset_path: str,
    bias_grid: List[float],
    output_prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads the census dataset, extracts DISTINCT domain of hours-per-week, and evaluates
    expected user utility for (1) babbling and (2) maximally informative at each bias level.
    Returns a DataFrame and optionally writes CSV/plots when output_prefix is provided.
    """
    s = load_census_hours_csv(dataset_path)
    domain_values = s.dropna().unique().tolist()
    domain = dedupe_and_sort_desc(domain_values)
    prior = PriorSpec(kind='uniform')

    rows = []
    for b in bias_grid:
        bias_obj = make_constant_bias(b)
        biases = biases_from_bias_obj(domain, bias_obj)

        # Babbling: single group
        q_babble = [domain[:]]  # one big group
        u_babble = expected_utility_for_query(q_babble, domain, prior, biases)

        # Maximally informative: Alg.2 + Alg.4
        q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, receiver_model='threshold')
        q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, receiver_model='threshold')
        u_star = expected_utility_for_query(q_star, domain, prior, biases)

        rows.append({'bias_level': b, 'utility_babbling': u_babble, 'utility_maxinfo': u_star})

    df = pd.DataFrame(rows).sort_values('bias_level')

    # Save outputs if desired
    if output_prefix:
        out_dir = Path(output_prefix).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = f"{output_prefix}results.csv"
        png_path = f"{output_prefix}plot.png"
        df.to_csv(csv_path, index=False)

        # Plot
        plt.figure()
        plt.plot(df['bias_level'], df['utility_babbling'], marker='o', label='Babbling baseline')
        plt.plot(df['bias_level'], df['utility_maxinfo'], marker='o', label='Maximally informative')
        plt.xlabel('Bias level (constant across domain)')
        plt.ylabel('Expected user utility (sum over domain)')
        plt.title('Babbling vs. Maximally Informative (Threshold; Uniform Prior)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"Wrote CSV to: {csv_path}")
        print(f"Wrote PNG to: {png_path}")

    return df

# ----------------------------
# Main
# ----------------------------

if __name__ == '__main__':
    # Adjust the dataset path to your local census CSV
    DATASET_PATH = '../data/real/census.csv'  # e.g., a CSV with a 'hours-per-week' column
    OUTPUT_PREFIX = '/Users/aryal/Desktop/Querying-COI/results/babble_vs_maxinfo_'

    # Bias levels to test (0..1). You can densify to e.g. np.linspace(0,1,41)
    bias_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run
    try:
        _ = run_babble_vs_maxinfo(DATASET_PATH, bias_levels, output_prefix=OUTPUT_PREFIX)
    except FileNotFoundError as e:
        print(f"Dataset not found at '{DATASET_PATH}'. Please set DATASET_PATH to your census CSV.")
