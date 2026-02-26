# -*- coding: utf-8 -*-
"""
Runtime scaling experiment (synthetic domain, sizes 2..1000).

- X-axis: domain size (distinct values)
- Y-axis: wall-clock time (milliseconds)
- Two lines: Algorithm 1 vs Whole pipeline (Alg.2 + Alg.4)
- Fixed: uniform prior, threshold receiver, top-10% high bias, mid-band medium bias
"""

import sys, time
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    algorithm_1_credibility_detection, run_pipeline_domain
)

# ----------------------------
# Bias profile (deterministic)
# ----------------------------
def make_top_mid_low_bias(
    high: float = 0.9, mid: float = 0.6, low: float = 0.1,
    top_cut: float = 0.9, mid_lo: float = 0.4, mid_hi: float = 0.6
) -> Bias1D:
    """Deterministic, size-agnostic bias over normalized t∈[0,1]."""
    def f(x: float, info: Dict[str, float]) -> float:
        lo, hi = float(info['min']), float(info['max'])
        t = 0.0 if hi == lo else (float(x) - lo) / (hi - lo)
        if t >= top_cut:
            return high        # top 10%
        if mid_lo <= t <= mid_hi:
            return mid         # mid band (e.g., 40-60%)
        return low             # elsewhere
    return Bias1D(kind='custom', custom=f)

# ----------------------------
# Timing helpers (milliseconds)
# ----------------------------
def time_algorithm_1(n: int, prior: PriorSpec, bias_obj: Bias1D) -> float:
    """Time Alg. 1 on a descending singleton grouping. Returns ms."""
    values = list(range(n))               # synthetic DISTINCT domain
    domain = dedupe_and_sort_desc(values) # descending, distinct
    biases = biases_from_bias_obj(domain, bias_obj)
    q_groups = [[v] for v in domain]      # singleton groups
    t0 = time.perf_counter()
    _ = algorithm_1_credibility_detection(q_groups, domain, prior, biases=biases, receiver_model='threshold')
    milliseconds = (time.perf_counter() - t0) * 1000.0
    return milliseconds

def time_pipeline(n: int, prior: PriorSpec, bias_obj: Bias1D) -> float:
    """Time the full pipeline (Alg.2 + Alg.4). Returns ms."""
    values = list(range(n))
    t0 = time.perf_counter()
    _ = run_pipeline_domain(values, prior, bias_obj, receiver_model='threshold', gamma=1.0)
    milliseconds = (time.perf_counter() - t0) * 1000.0
    return milliseconds

# ----------------------------
# Experiment runner
# ----------------------------
def run_experiment(sizes: List[int]) -> pd.DataFrame:
    prior = PriorSpec(kind='uniform')
    bias_obj = make_top_mid_low_bias(high=0.9, mid=0.6, low=0.1, top_cut=0.9, mid_lo=0.4, mid_hi=0.6)

    rows = []
    for n in sizes:
        t_alg1 = time_algorithm_1(n, prior, bias_obj)
        t_pipe = time_pipeline(n, prior, bias_obj)
        rows.append({'domain_size': n, 'time_alg1_ms': t_alg1, 'time_pipeline_ms': t_pipe})
    df = pd.DataFrame(rows).sort_values('domain_size')
    return df

# ----------------------------
# Plotting
# ----------------------------
def _fit_power_loglog(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope p in log–log fit y ~ c * x^p (for diagnostics)."""
    m = (x > 0) & (y > 0)
    p, _ = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(p)

def plot_results(df: pd.DataFrame, png_linear: str, png_loglog: str) -> None:
    # Linear scale
    plt.figure()
    plt.plot(df['domain_size'], df['time_alg1_ms'], label='Algorithm 1')
    plt.plot(df['domain_size'], df['time_pipeline_ms'], label='Pipeline (Alg.2+4)')
    plt.xlabel('Domain size')
    plt.ylabel('Time (milliseconds)')
    plt.title('Runtime vs Domain Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_linear, dpi=150)
    plt.close()

    # Log–log (useful to see polynomial order)
    p1 = _fit_power_loglog(df['domain_size'].to_numpy(), df['time_alg1_ms'].to_numpy())
    p2 = _fit_power_loglog(df['domain_size'].to_numpy(), df['time_pipeline_ms'].to_numpy())
    plt.figure()
    plt.loglog(df['domain_size'], df['time_alg1_ms'], label=f'Credibility Detection (slope≈{p1:.2f})')
    plt.loglog(df['domain_size'], df['time_pipeline_ms'], label=f'Maximally Informative (slope≈{p2:.2f})')
    plt.xlabel('Domain size (log)')
    plt.ylabel('Time (ms, log)')
    plt.title('Runtime vs Domain Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_loglog, dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Adjust this to your preferred folder
    PATH_PREFIX = Path('/Users/aryal/Desktop/Querying-COI/results/runtime_scaling_')
    PATH_PREFIX.parent.mkdir(parents=True, exist_ok=True)

    # Choose sizes (use a larger step locally if you go up to 1000)
    sizes = list(range(2, 1000, 50))  # e.g., 2,7,12,...,97

    df = run_experiment(sizes)
    csv_path = str(PATH_PREFIX) + 'sample.csv'
    png_linear = str(PATH_PREFIX) + 'linear.png'
    png_loglog = str(PATH_PREFIX) + 'loglog.png'

    df.to_csv(csv_path, index=False)
    plot_results(df, png_linear, png_loglog)

    # Optional: also print fitted slopes
    p1 = _fit_power_loglog(df['domain_size'].to_numpy(), df['time_alg1_ms'].to_numpy())
    p2 = _fit_power_loglog(df['domain_size'].to_numpy(), df['time_pipeline_ms'].to_numpy())
    print(f"Wrote CSV to: {csv_path}")
    print(f"Wrote linear plot to: {png_linear}")
    print(f"Wrote log–log plot to: {png_loglog}")
    print(f"Estimated slopes (log–log): Alg.1≈{p1:.2f}, Pipeline≈{p2:.2f}")
    print(df)
