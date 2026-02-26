# -*- coding: utf-8 -*-
"""
Asymmetry sweep (ONE dataset / ONE column):
Babbling vs Maximally Informative (Threshold; Uniform Prior)
+ OPTIONAL: Start from a user-provided PARTIAL RANKING (ties allowed).

What this file does:
- Loads a numeric column, builds the DISTINCT domain (DESC).
- Builds per-value bias via a sigmoid (anchored or min-max normalized).
- Evaluates:
    • Uninformative baseline (single block)
    • Algorithmic pipeline starting from:
        (a) strict order (singletons)      – default
        (b) user partial ranking (ties)     – if provided (bins or explicit groups)
- Returns a tidy DataFrame per k_slope with utilities & asymmetry metric.
- Plots are provided by a helper (no seaborn, matplotlib only).

Minimal changes from your original:
- Added an import for the group-respecting A2 function.
- Added helpers to accept user bins or explicit groups.
- Added an optional output column 'utility_user_query' when q_user is provided.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make local module importable if needed
if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    compute_expected_posteriors, system_best_response,
    algorithm_2_build_qbase, algorithm_4_maximally_informative,
    algorithm_2_build_qbase_from_groups,  # NEW: start from partial rankings
)

# -----------------------------------------------------------------------------
# Bias & metrics
# -----------------------------------------------------------------------------

def make_sigmoid_asym_bias(
    k_slope: float,
    *,
    k_zero: float = 0.0,
    center: Optional[float] = None,   # optional value-anchored center (orig units)
    scale: Optional[float] = None     # optional value-anchored scale  (orig units)
) -> Bias1D:
    """
    Value-specific asymmetric bias.
    - Default: normalize to t∈[0,1] and use b(v) = σ(k_slope*(t - 0.5))
    - Optional: anchor in original units with center/scale, b(v) = σ(k_slope * ((v - center)/scale))
    - If k_slope <= k_zero, return 0.5 everywhere (no asymmetry).
    """
    def f(x: float, info: Dict[str, float]) -> float:
        if k_slope <= k_zero:
            return 0.5
        if center is not None and scale is not None and scale > 0:
            z = (float(x) - center) / scale
            return 1.0 / (1.0 + np.exp(-k_slope * z))
        lo, hi = float(info['min']), float(info['max'])
        if hi == lo:
            return 0.5
        t = (float(x) - lo) / (hi - lo)
        return 1.0 / (1.0 + np.exp(-k_slope * (t - 0.5)))
    return Bias1D(kind='custom', custom=f)


def compute_asymmetry_metric(domain: List[float], bias_map: Dict[float, float]) -> float:
    """
    Δ = mean_bias(top half by value) − mean_bias(bottom half by value).
    A quick scalar that increases with value-wise asymmetry.
    """
    n = len(domain)
    if n == 0:
        return 0.0
    asc = sorted(domain)
    half = n // 2
    bottom, top = asc[:half], asc[half:]
    mbot = float(np.mean([bias_map[v] for v in bottom])) if bottom else 0.0
    mtop = float(np.mean([bias_map[v] for v in top])) if top else 0.0
    return mtop - mbot

# -----------------------------------------------------------------------------
# Utility for evaluation (threshold model)
# -----------------------------------------------------------------------------

def expected_utility_for_query(
    q_groups: List[List[float]],
    domain: List[float],
    prior: PriorSpec,
    biases: Dict[float, float]
) -> float:
    """
    Normalized threshold utility:
        U_norm = (∑_v θ[v] * keep[v]) / (∑_v θ[v])
    where keep[v] = 1{posterior[v] > bias[v]} under the query q_groups.
    """
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta  = system_best_response(q_groups, domain, prior, biases, receiver_model='threshold')
    num = sum(theta.get(v, 0.0) * beta.get(v, 0) for v in theta.keys())
    den = sum(theta.values())
    return 0.0 if den <= 0.0 else float(num / den)

# -----------------------------------------------------------------------------
# Data loading (one dataset / one column)
# -----------------------------------------------------------------------------

def load_numeric_series(dataset_path: str, column: str) -> pd.Series:
    """Load the requested numeric column (drops NaNs)."""
    df = pd.read_csv(dataset_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {dataset_path}.")
    s = pd.to_numeric(df[column], errors='coerce').dropna()
    if s.empty:
        raise ValueError(f"Column '{column}' has no numeric values.")
    return s

# -----------------------------------------------------------------------------
# Optional user partial ranking: groups via bins
# -----------------------------------------------------------------------------

def groups_from_bins_desc(
    domain_desc: List[float],
    bins: List[Tuple[float, Optional[float]]],
) -> List[List[float]]:
    """
    Build user groups from numeric bins over the DISTINCT domain (DESC).
    - Bins are [lo, hi); hi=None means [lo, +inf).
    - Keeps DESC domain order inside each group (no re-sorting).
    """
    out: List[List[float]] = []
    for lo, hi in bins:
        S = set(v for v in domain_desc if (v >= lo) and (hi is None or v < hi))
        grp = [v for v in domain_desc if v in S]
        if grp:
            out.append(grp)
    return out

# -----------------------------------------------------------------------------
# Runner (one dataset)
# -----------------------------------------------------------------------------

def run_asymmetry_sweep_one(
    dataset_path: str,
    column: str,
    k_grid: List[float],
    *,
    k_zero: float = 0.0,
    center: Optional[float] = None,
    scale: Optional[float] = None,
    use_distinct_domain: bool = True,
    save_csv: Optional[str] = None,
    # NEW (optional): user-provided partial ranking as bins or explicit groups
    user_bins: Optional[List[Tuple[float, Optional[float]]]] = None,
    q_user_groups: Optional[List[List[float]]] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per k in k_grid:
      [asymmetry_delta, k_slope, utility_babbling, utility_maxinfo, (optional) utility_user_query]

    Notes:
    - Domain is DISTINCT values when use_distinct_domain=True (recommended).
    - If user_bins or q_user_groups is provided, we HONOR the partial ranking and
      run Alg. 2 from groups; otherwise we start from singletons as before.
    """
    s = load_numeric_series(dataset_path, column)
    values = s.unique().tolist() if use_distinct_domain else s.tolist()
    domain = dedupe_and_sort_desc(values)
    prior = PriorSpec(kind='uniform')

    # Optional: construct a user partial ranking
    q_user = None
    if q_user_groups is not None:
        q_user = q_user_groups
        flat = [x for g in q_user for x in g]
        if set(flat) != set(domain) or len(flat) != len(domain):
            raise ValueError("q_user_groups must be a partition of the domain (no dups/missing).")
    elif user_bins is not None:
        q_user = groups_from_bins_desc(domain, user_bins)

    rows = []
    print(f"[sweep] Loaded {len(domain)} distinct values from '{dataset_path}', column '{column}'.")

    for k in k_grid:
        # Build per-value bias for this k
        bias_obj = make_sigmoid_asym_bias(k_slope=k, k_zero=k_zero, center=center, scale=scale)
        biases = biases_from_bias_obj(domain, bias_obj)
        delta = compute_asymmetry_metric(domain, biases)

        # Uninformative baseline: single block over the whole domain
        q_babble = [domain[:]]
        u_babble = expected_utility_for_query(q_babble, domain, prior, biases)

        # Build q_base:
        # - If q_user is provided, MERGE-only from q_user (honor ties).
        # - Else, start from singletons (original behavior).
        if q_user is None:
            q_base = algorithm_2_build_qbase(
                domain, domain, prior, biases=biases, receiver_model='threshold', gamma=1.0, eps_order=1e-6
            )
        else:
            q_base = algorithm_2_build_qbase_from_groups(
                q_user, domain, prior, biases=biases, receiver_model='threshold', gamma=1.0, eps_order=1e-6
            )

        # Maximally informative q_star (Alg. 4)
        q_star = algorithm_4_maximally_informative(
            q_base, domain, prior, biases=biases, receiver_model='threshold', gamma=1.0, eps_order_for_tiebreak=1e-12
        )

        # Utilities
        u_star = expected_utility_for_query(q_star, domain, prior, biases)
        u_user = expected_utility_for_query(q_user, domain, prior, biases) if q_user is not None else np.nan

        print(f"[sweep] k={k:>4}  Δ={delta:+.4f}  q_babble={len(q_babble)}g  q_base={len(q_base)}g  q_star={len(q_star)}g  "
              f"U_un={u_babble:.4f}  U_star={u_star:.4f}{'' if np.isnan(u_user) else f'  U_user={u_user:.4f}'}")

        rows.append({
            'asymmetry_delta': float(delta),
            'k_slope': float(k),
            'utility_babbling': float(u_babble),
            'utility_maxinfo': float(u_star),
            'utility_user_query': float(u_user) if not np.isnan(u_user) else np.nan,  # NEW column (optional)
        })

    df = pd.DataFrame(rows).sort_values('asymmetry_delta').reset_index(drop=True)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"[sweep] Wrote CSV: {save_csv}")
    return df

# -----------------------------------------------------------------------------
# Plotting (separate, reusable; matplotlib only)
# -----------------------------------------------------------------------------

def plot_asymmetry_df(
    df: pd.DataFrame,
    *,
    label: str = '',
    output_png: Optional[str] = None,
    normalize_each_curve: bool = False
) -> None:
    """
    Plot utilities vs asymmetry_delta from a single sweep DataFrame.
    - Solid line: Max-Info; Dashed line: Un-Info
    - Optionally normalize each curve to [0,1] to compare shapes only.
    """
    if df.empty:
        print("[plot] Empty DataFrame; nothing to plot.")
        return

    x = df['asymmetry_delta'].astype(float).values
    y_bab = df['utility_babbling'].astype(float).values
    y_mxi = df['utility_maxinfo'].astype(float).values

    if normalize_each_curve:
        def _norm(y: np.ndarray) -> np.ndarray:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            return (y - y_min) / (y_max - y_min) if y_max > y_min else y * 0.0
        y_bab = _norm(y_bab)
        y_mxi = _norm(y_mxi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_mxi, linewidth=2.6, linestyle='-',  marker='o', label=f'{label} — Max-Info')
    plt.plot(x, y_bab, linewidth=2.2, linestyle='--', marker='x', label=f'{label} — Un-Info')
    plt.xlabel('Asymmetry Δ (mean bias top-half − bottom-half)')
    plt.ylabel('Expected User Utility (normalized)')
    ttl = 'Utility vs Bias Asymmetry (Threshold; Uniform Prior)'
    if label:
        ttl += f' — {label}'
    plt.title(ttl)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, dpi=150)
        print(f"[plot] Wrote PNG: {output_png}")
        plt.close()
    else:
        plt.close()

# -----------------------------------------------------------------------------
# Example usage (two modes)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Example: adjust to your file & column
    DATASET_PATH = '../data/real/COMPAS.csv'
    COLUMN_NAME  = 'age'  # numeric column with enough distinct values
    LABEL        = f'COMPAS: {COLUMN_NAME}'

    # Include k=0 to see Max-Info ≈ Un-Info at symmetry
    k_values = [0, 1, 2, 3, 4, 6, 8, 10, 12]

    # (A) Strict order pipeline (original behavior)
    df_strict = run_asymmetry_sweep_one(
        DATASET_PATH, COLUMN_NAME, k_values,
        k_zero=0.0, center=None, scale=None, use_distinct_domain=True,
        save_csv=f'../results/asymmetry_sweep_{COLUMN_NAME}_strict.csv'
    )
    plot_asymmetry_df(
        df_strict, label=LABEL + " (strict)",
        output_png=f'../results/asymmetry_sweep_{COLUMN_NAME}_strict.png',
        normalize_each_curve=False
    )

    # (B) Start from a user-provided PARTIAL RANKING (e.g., age buckets)
    #     (ties intact; we will only MERGE boundaries if unsupported)
    df_tied = run_asymmetry_sweep_one(
        DATASET_PATH, COLUMN_NAME, k_values,
        k_zero=0.0, center=None, scale=None, use_distinct_domain=True,
        save_csv=f'../results/asymmetry_sweep_{COLUMN_NAME}_tied.csv',
        user_bins=[(0,18), (18,30), (30,60), (60, None)]   # partial ranking by age bins
        # Alternatively: pass q_user_groups=[[...],[...],...] covering the full domain.
    )
    plot_asymmetry_df(
        df_tied, label=LABEL + " (tied start)",
        output_png=f'../results/asymmetry_sweep_{COLUMN_NAME}_tied.png',
        normalize_each_curve=False
    )
