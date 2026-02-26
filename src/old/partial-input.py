# -*- coding: utf-8 -*-
"""
Pipeline runner that ACCEPTS a PARTIAL RANKING as input (ties allowed)
and runs: Alg.1 (credibility on OP(q_user)) -> Alg.2 (group-respecting build_qbase)
-> Alg.4 (maximally informative), with per-value biases.

Relies on your existing `coi_algorithms.py`:
  PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
  compute_expected_posteriors, system_best_response,
  algorithm_1_credibility_detection, algorithm_4_maximally_informative,
  user_utility_from_response  (optional for extra metrics)

Usage:
  - Provide domain and q_user directly, OR load from a dataset column and bin it.
  - See the __main__ section at the bottom for two runnable examples.
"""

from __future__ import annotations
import sys
from typing import List, Dict, Hashable, Iterable, Tuple, Optional
from collections import defaultdict, deque

# Make local module importable if needed
if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    compute_expected_posteriors, system_best_response,
    algorithm_1_credibility_detection, algorithm_4_maximally_informative,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

Value = Hashable

def flatten_in_order(q_groups: List[List[Value]]) -> List[Value]:
    """Concatenate groups preserving group order and intra-group order."""
    out: List[Value] = []
    for G in q_groups:
        out.extend(G)
    return out

def op_pairs_strict(q_groups: List[List[Value]]) -> List[Tuple[Value, Value]]:
    """
    Strict ordered pairs present in the query:
      all (u,v) with u in an earlier group than v.
    No intra-group pairs.
    """
    pairs = []
    for i in range(len(q_groups)):
        for j in range(i+1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def build_reachability(credible_pairs: Iterable[Tuple[Value, Value]]) -> Dict[Value, set]:
    """
    Build forward reachability (transitive closure) over the directed graph defined by C.
    """
    adj: Dict[Value, List[Value]] = defaultdict(list)
    nodes: set = set()
    for u, v in credible_pairs:
        adj[u].append(v)
        nodes.add(u); nodes.add(v)

    # BFS from each node (graph is small, this is fine)
    reach: Dict[Value, set] = {x: set() for x in nodes}
    for s in nodes:
        dq = deque([s])
        seen = {s}
        while dq:
            cur = dq.popleft()
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    dq.append(nxt)
                    reach[s].add(nxt)
    return reach

def expected_utility_threshold(
    q_groups: List[List[Value]],
    domain: List[Value],
    prior: PriorSpec,
    biases: Dict[Value, float],
) -> float:
    """
    Normalized user utility under the THRESHOLD receiver:
        U_norm = (∑ θ[v] * keep[v]) / (∑ θ[v]),
    where keep[v] = 1{posterior[v] > bias[v]} computed from q_groups.
    """
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta  = system_best_response(q_groups, domain, prior, biases, receiver_model='threshold')
    num = sum(theta.get(v, 0.0) * float(beta.get(v, 0)) for v in theta.keys())
    den = sum(theta.values())
    return 0.0 if den <= 0.0 else float(num / den)

# -----------------------------------------------------------------------------
# Algorithm 2 (group-respecting variant): start from q_user groups, MERGE only
# -----------------------------------------------------------------------------

def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[Value]],
    domain: List[Value],
    prior: PriorSpec,
    *,
    biases: Dict[Value, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order: float = 1e-9,
    verbose: bool = False,
) -> List[List[Value]]:
    """
    Group-respecting A2:
      - starts from the user-provided partial ranking q_init_groups (ties intact),
      - repeatedly MERGES adjacent groups whose boundary is NOT supported by the
        transitive closure of credible edges C returned by Alg. 1 on the current q,
      - NEVER splits ties (consistent with partial-ranking-as-strategy).

    This mirrors your A2 logic but honors grouped input.
    """
    q_cur = [g[:] for g in q_init_groups]  # shallow copy of groups

    # To be safe, ensure that groups cover the domain exactly (no duplicates/missing)
    flat = flatten_in_order(q_cur)
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_init_groups must be a partition of `domain` in order.")

    # Main merge loop — bounded by (#groups - 1) iterations
    for _ in range(max(0, len(q_cur) - 1) + 1):
        # 1) Build credibility set C for the current grouped query
        C: List[Tuple[Value, Value]] = algorithm_1_credibility_detection(
            q_cur, domain, prior,
            biases=biases, receiver_model=receiver_model,
            gamma=gamma, eps_order=eps_order
        )

        # 2) Compute reachability over C
        reach = build_reachability(C)

        # 3) Find the first unsupported adjacent boundary and MERGE
        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i+1]
            supported = True
            for u in Gi:
                for v in Gj:
                    # Boundary Gi -> Gj is supported only if v is reachable from u in C
                    if v not in reach.get(u, set()):
                        supported = False
                        break
                if not supported:
                    break
            if not supported:
                if verbose:
                    print(f"[A2] merge boundary at groups {i}|{i+1}: {Gi} + {Gj}")
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i+2:]
                merged = True
                break
            i += 1

        if not merged:
            # All adjacent boundaries are supported → fixed point
            break

    return q_cur

# -----------------------------------------------------------------------------
# Binning helper (for examples like age buckets)
# -----------------------------------------------------------------------------

def groups_from_bins_desc(
    domain_desc: List[Value],
    bins: List[Tuple[float, Optional[float]]],
) -> List[List[Value]]:
    """
    Construct groups from numeric bins while preserving the domain order (DESC).
    Bins are [lo, hi); hi=None means open-ended [lo, +inf).
    """
    out: List[List[Value]] = []
    for lo, hi in bins:
        S = set(v for v in domain_desc if (v >= lo) and (hi is None or v < hi))
        group = [v for v in domain_desc if v in S]
        if group:
            out.append(group)
    return out

# -----------------------------------------------------------------------------
# Top-level pipeline
# -----------------------------------------------------------------------------

def run_pipeline_partial_query(
    q_user: List[List[Value]],
    domain: List[Value],
    prior: PriorSpec,
    biases: Dict[Value, float],
    *,
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order: float = 1e-9,
    verbose: bool = True,
):
    """
    Run the full pipeline given a user-specified PARTIAL RANKING q_user.

    Returns a dict with:
      - q_user
      - credible_pairs (Alg.1 results on q_user)
      - q_base (group-respecting A2 starting from q_user)
      - q_star (Alg.4 starting from q_base)
      - utilities: dict of expected utilities for q_user, q_base, q_star (threshold)
    """
    # Sanity checks
    flat = flatten_in_order(q_user)
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_user must be a partition of `domain` (cover exactly once).")

    # Alg. 1 on the user partial ranking
    C = algorithm_1_credibility_detection(
        q_user, domain, prior,
        biases=biases, receiver_model=receiver_model,
        gamma=gamma, eps_order=eps_order
    )
    if verbose:
        print(f"[Alg1] |OP(q_user)|={len(op_pairs_strict(q_user))}, |C|={len(C)}")

    # Alg. 2 (group-respecting) starting from q_user
    q_base = algorithm_2_build_qbase_from_groups(
        q_user, domain, prior,
        biases=biases, receiver_model=receiver_model,
        gamma=gamma, eps_order=eps_order, verbose=verbose
    )
    if verbose:
        print(f"[Alg2] q_base has {len(q_base)} groups")

    # Alg. 4 starting from q_base
    q_star = algorithm_4_maximally_informative(
        q_base, domain, prior,
        biases=biases, receiver_model=receiver_model,
        gamma=gamma, eps_order_for_tiebreak=1e-12
    )
    if verbose:
        print(f"[Alg4] q_star has {len(q_star)} groups")

    # Utilities (threshold model)
    util_q_user = expected_utility_threshold(q_user, domain, prior, biases)
    util_q_base = expected_utility_threshold(q_base, domain, prior, biases)
    util_q_star = expected_utility_threshold(q_star, domain, prior, biases)
    utilities = {
        "q_user": util_q_user,
        "q_base": util_q_base,
        "q_star": util_q_star,
    }
    if verbose:
        print("[Util] q_user={:.4f}, q_base={:.4f}, q_star={:.4f}".format(
            util_q_user, util_q_base, util_q_star))

    return {
        "q_user": q_user,
        "credible_pairs": C,
        "q_base": q_base,
        "q_star": q_star,
        "utilities": utilities,
    }

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # EXAMPLE A: toy domain with per-value biases and a tie
    domain = ['e1', 'e2', 'e3', 'e4']   # DESC order expected by your codebase
    q_user = [['e1'], ['e2','e3'], ['e4']]  # partial ranking e1 < {e2~e3} < e4
    prior = PriorSpec(kind='uniform')
    biases = {'e1': 0.8, 'e2': 0.0, 'e3': 0.8, 'e4': 0.3}  # per-value biases

    print("\n=== TOY EXAMPLE ===")
    result = run_pipeline_partial_query(
        q_user, domain, prior, biases,
        receiver_model="threshold", gamma=1.0, eps_order=1e-9, verbose=True
    )

    # EXAMPLE B: age bins (0–18, 18–30, 30–60, 60+) with per-value bias from a sigmoid
    try:
        import pandas as pd
        from pathlib import Path

        # Load your dataset/column
        DATASET_PATH = '../data/real/COMPAS.csv'
        COLUMN_NAME  = 'age'
        df = pd.read_csv(DATASET_PATH)
        s = pd.to_numeric(df[COLUMN_NAME], errors='coerce').dropna()
        values = s.unique().tolist()
        domain_desc = dedupe_and_sort_desc(values)  # match domain-level semantics

        # Build partial ranking via bins
        bins = [(0,18), (18,30), (30,60), (60, None)]
        q_bins = groups_from_bins_desc(domain_desc, bins)

        # Bias per value (use your Bias1D factory if you want)
        def bias_sigmoid(center: float, scale: float, k: float):
            import numpy as np
            def f(x: float, info: Dict[str, float]) -> float:
                z = (float(x) - center) / scale
                return 1.0 / (1.0 + np.exp(-k * z))
            return Bias1D(kind='custom', custom=f)

        prior2 = PriorSpec(kind='uniform')
        bias_obj = bias_sigmoid(center=40.0, scale=12.0, k=4.0)
        biases2 = biases_from_bias_obj(domain_desc, bias_obj)

        print("\n=== AGE BIN EXAMPLE ===")
        result_bins = run_pipeline_partial_query(
            q_bins, domain_desc, prior2, biases2,
            receiver_model="threshold", gamma=1.0, eps_order=1e-9, verbose=True
        )
    except Exception as e:
        print(f"[info] Skipped age-bin example due to: {e}")
