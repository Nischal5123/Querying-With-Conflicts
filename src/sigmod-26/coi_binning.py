# -*- coding: utf-8 -*-
"""
Conservative (no false-positives) bin-level credibility for COI (uniform prior + threshold receiver)

What this script provides
-------------------------
1) Original full-domain algorithms:
   - Alg.2  : Build q_base by merging unsupported boundaries (using Alg.1 edges)
   - Alg.4  : Maximally-informative DP over q_base
   - evaluate_plan_utility(...)

2) Conservative bin-only Alg.2:
   - Uses a bin-local interval-sufficient check per boundary (no Alg.1 edges).
   - No false positives: if a boundary is kept at bin-level, then every item-pair
     across it is credible at item-level (for uniform-prior + threshold receiver).
   - Conservative: may miss some item-level credible pairs ⇒ utility can be lower
     than full domain, but never higher. As bins get narrower, utility is non-decreasing.

3) Experiments:
   - Sweep number of bins for a fixed domain, write a CSV of utilities/timings.
   - Includes a "full" line (one item per bin) for comparison.

Notes
-----
- Utilities are computed in the existing framework: Θ are expected posteriors in [0,1].
- This file focuses on the threshold receiver with uniform prior (the setting
  where the conservative test is valid and tightest).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
import itertools, json, os, time, math, csv as _csv

import numpy as np

# ---------------------------------------------------------------------
# Global tolerances
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-9    # numeric comparison tolerance
EPS_ORDER   = 0.0     # presentation epsilon not used here


# =============================================================================
# Priors (how rank maps to expected preference)
# =============================================================================

@dataclass
class PriorSpec:
    """
    Prior over permutation ranks. rank_expectation(k, r_desc) returns E[Θ] for the
    item at DESCENDING rank r_desc among k distinct values (r_desc = 1 is the top).
    """
    kind: str = "uniform"
    a: float = 1.0
    b: float = 1.0
    lam: float = 1.0
    p: float = 1.0
    custom: Optional[Callable[[int, int], float]] = None

    def rank_expectation(self, k: int, r_desc: int) -> float:
        """Expected preference for DESC rank r_desc among k, based on prior kind."""
        if k <= 0 or r_desc <= 0 or r_desc > k:
            return 0.0

        if self.kind == "uniform":
            j = k - r_desc + 1  # j = rank in ASC order
            return j / (k + 1.0)

        if self.kind == "beta":
            j = k - r_desc + 1
            den = k + self.a + self.b - 1.0
            if den <= 0:
                return 0.0
            return float(min(max((j + self.a - 1.0) / den, 0.0), 1.0))

        if self.kind == "exp_kernel":
            ws = [math.exp(-self.lam * (rr - 1)) for rr in range(1, k + 1)]
            wmax, wmin = max(ws), min(ws)
            if wmax == wmin:
                return 1.0
            return (ws[r_desc - 1] - wmin) / (wmax - wmin + EPS_COMPARE)

        if self.kind == "power_kernel":
            ws = [1.0 / (rr ** self.p) for rr in range(1, k + 1)]
            wmax, wmin = max(ws), min(ws)
            if wmax == wmin:
                return 1.0
            return (ws[r_desc - 1] - wmin) / (wmax - wmin + EPS_COMPARE)

        if self.kind == "custom" and self.custom:
            v = float(self.custom(k, r_desc))
            return float(min(max(v, 0.0), 1.0))

        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(
    q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec
) -> Dict[Any, float]:
    """
    Compute expected posteriors for each domain value GIVEN the group ranks in q_groups.
    - q_groups is a list of groups (ties allowed).
    - All items in a group share the same posterior: the average of the expectations
      across the consecutive ranks the group occupies.
    - domain (DESC) defines k and (indirectly) the rank spectrum.
    Works for non-numeric domain values (e.g., tuples).
    """
    k = len(domain)
    post: Dict[Any, float] = {}
    r = 1  # current group’s starting DESC rank

    for g in q_groups:
        n = len(g)
        if n == 0:
            continue

        if prior.kind == "uniform":
            j_hi = k - r + 1
            j_lo = k - (r + n - 1) + 1
            a = ((j_hi + j_lo) / 2.0) / (k + 1.0)
        else:
            a = sum(prior.rank_expectation(k, rr) for rr in range(r, r + n)) / float(n)

        for v in g:
            post[v] = float(a)
        r += n

    for v in domain:
        post.setdefault(v, 0.0)
    return post


# =============================================================================
# Receiver models & utilities
# =============================================================================

def system_best_response_threshold(
    q_groups: List[List[Any]], domain: List[Any],
    post: Dict[Any, float], biases: Dict[Any, float],
) -> Dict[Any, int]:
    """Threshold receiver: keep(v) = 1{post[v] > bias[v]}."""
    return {v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0
            for g in q_groups for v in g}

def system_best_response(
    q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec,
    biases: Dict[Any, float], receiver_model: str = "threshold"
):
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    raise ValueError("This file implements the threshold receiver variant.")

def user_utility_from_response(
    theta: Dict[Any, float], response: Dict[Any, int]
) -> float:
    return sum(theta.get(v, 0.0) * int(response.get(v, 0)) for v in theta.keys())

def evaluate_plan_utility(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any, float],
) -> Tuple[float, int]:
    """
    Utility uses θ computed on singleton domain (no presentation epsilon).
    """
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta  = system_best_response(q_groups, domain, prior, biases, receiver_model="threshold")
    util  = user_utility_from_response(theta, beta)
    kept  = sum(int(beta.get(v, 0)) for v in (x for g in q_groups for x in g))
    return float(util), kept


# =============================================================================
# Original Alg.2 (uses Alg.1 edges) + Alg.4 (unchanged)
# =============================================================================

def op_pairs_strict(q_groups: List[List[Any]]) -> List[Tuple[Any, Any]]:
    pairs: List[Tuple[Any, Any]] = []
    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def _reachability_bitsets(items: List[Any], edges: Set[Tuple[Any, Any]]):
    n = len(items)
    idx = {v: i for i, v in enumerate(items)}
    neigh = [[] for _ in range(n)]
    for u, v in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is not None and iv is not None and iu < iv:
            neigh[iu].append(iv)
    R = [0] * n
    for i in range(n - 1, -1, -1):
        mask = 0
        for j in neigh[i]:
            mask |= (1 << j) | R[j]
        R[i] = mask
    return idx, R

# def _bin_boundary_supported_interval_sufficient_envelope(
#     Gi: List[Any], Gj: List[Any],
#     *, idx: Dict[Any, int], k: int, prior: PriorSpec, biases: Dict[Any, float]
# ) -> bool:
#     """
#     Envelope-safe check for boundary Gi|Gj (Gi above Gj, domain is DESC).
#     Uses worst-case ambiguous interval [theta_high(Gj), theta_low(Gi)) so there are NO false positives.
#     """
#     # Domain ranks (DESC) are contiguous within each bin since we only merge adjacent bins.
#     i_start = min(idx[x] for x in Gi)  # 0-based index in domain
#     i_end   = max(idx[x] for x in Gi)
#     j_start = min(idx[x] for x in Gj)
#     j_end   = max(idx[x] for x in Gj)
#
#     # Convert to DESC ranks r_desc = index+1
#     r_i_low  = i_end   + 1  # lowest theta in Gi
#     r_j_high = j_start + 1  # highest theta in Gj
#
#     theta_i_low  = prior.rank_expectation(k, r_i_low)
#     theta_j_high = prior.rank_expectation(k, r_j_high)
#
#     # If the envelope collapses or flips, nothing ambiguous remains
#     if theta_i_low <= theta_j_high + EPS_COMPARE:
#         return True
#
#     Bi_min = min(biases[x] for x in Gi); Bi_max = max(biases[x] for x in Gi)
#     Bj_min = min(biases[x] for x in Gj); Bj_max = max(biases[x] for x in Gj)
#
#     # V-safe: all of Gj outside the interval
#     vsafe = (Bj_max < theta_j_high - EPS_COMPARE) or (Bj_min >= theta_i_low + EPS_COMPARE)
#     # U-safe: all of Gi inside the interval
#     usafe = (Bi_min >= theta_j_high + EPS_COMPARE) and (Bi_max < theta_i_low - EPS_COMPARE)
#
#     return vsafe or usafe

def _boundary_supported_by_edges(idx_map, R, Gi: List[Any], Gj: List[Any],
                                 *, policy: str = "all", alpha: float = 0.5) -> bool:
    """
    policy:
      - "all":      every (u,v) with u∈Gi, v∈Gj is reachable
      - "any":      at least one (u,v) is reachable
      - "quantile": at least alpha fraction of pairs are reachable (0<alpha<=1)
    """
    if not Gi or not Gj:
        return True

    want_mask = 0
    for v in Gj:
        want_mask |= (1 << idx_map[v])

    if policy == "all":
        for u in Gi:
            if (R[idx_map[u]] & want_mask) != want_mask:
                return False
        return True

    total = len(Gi) * len(Gj)
    supported = 0
    for u in Gi:
        mask_u = R[idx_map[u]]
        if policy == "any":
            if mask_u & want_mask:
                return True
        else:  # "quantile"
            for v in Gj:
                if (mask_u >> idx_map[v]) & 1:
                    supported += 1

    if policy == "any":
        return False
    need = math.ceil(alpha * total)
    return supported >= need

def algorithm_1_credibility_detection_fast_uniform_threshold(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any, float],
) -> Set[Tuple[Any, Any]]:
    """
    Fast Alg.1 (uniform prior + threshold receiver; eps=0).
    A strict pair (u in Gi, v in Gj, i<j) is non-credible iff:
      (k_j(v)=0, k_i(v)=1) and (k_j(u)=k_i(u)  ⇒  k_j(u)-k_i(u)=0)
    i.e., v lies in [a_j, a_i) while u does NOT lie in [a_j, a_i).
    """
    items = [x for g in q_groups for x in g]
    k = len(domain)
    idx = {v: i for i, v in enumerate(items)}

    # group posterior levels
    a_g = []
    r = 1
    for g in q_groups:
        n = len(g)
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a_g.append(((j_hi + j_lo) / 2.0) / (k + 1.0))
        r += n

    C: Set[Tuple[Any, Any]] = set()
    for gi in range(len(q_groups)):
        for gj in range(gi + 1, len(q_groups)):
            ai, aj = a_g[gi], a_g[gj]
            for u in q_groups[gi]:
                bu = biases[u]
                # d_u = k_j(u) - k_i(u) in {0, -1}; equals -1 iff bu in [aj, ai)
                du = -1 if (bu + EPS_COMPARE >= aj and bu < ai - EPS_COMPARE) else 0
                for v in q_groups[gj]:
                    bv = biases[v]
                    # d_v = k_i(v) - k_j(v) in {0, 1}; equals 1 iff bv in [aj, ai)
                    dv =  1 if (bv + EPS_COMPARE >= aj and bv < ai - EPS_COMPARE) else 0
                    # non-credible only when du=0 and dv=1
                    if not (du == 0 and dv == 1):
                        C.add((u, v))
    return C

def algorithm_2_build_qbase(
    initial_order: List[Any], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[Any]]:
    """
    Full Alg.2 driven by Alg.1 edges (fast path for uniform+threshold).
    """
    q_cur: List[List[Any]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection_fast_uniform_threshold(
            q_cur, domain, prior, biases=biases
        )
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported_by_edges(idx_map, R, Gi, Gj,
                                                policy=boundary_policy, alpha=alpha):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur

def algorithm_4_maximally_informative(
    q_base: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any, float],
) -> List[List[Any]]:
    """
    Alg. 4: DP over contiguous runs of q_base to select merges that yield
    strictly positive base utility gains (threshold receiver).
    """
    m = len(q_base)
    if m == 0:
        return []

    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    # Precompute "baseline" block utilities for q_base
    beta_base = system_best_response(q_base, domain, prior, biases)
    base_gain0: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        u0 = sum(theta[x] * float(beta_base.get(x, 0)) for x in items_t)
        base_gain0.append(u0)

    # Build marginal gain table C0
    C0 = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            run_items = [x for g in q_base[i:j+1] for x in g]
            temp_q = q_base[:i] + [run_items] + q_base[j+1:]
            beta_run = system_best_response(temp_q, domain, prior, biases)
            u0_run = sum(theta[x] * float(beta_run.get(x, 0)) for x in run_items)
            base_line0 = sum(base_gain0[t] for t in range(i, j+1))
            C0[i][j] = u0_run - base_line0

    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    # DP: prefer higher base; break ties by later cuts (right-biased)
    dp   = [0.0] * (m + 1)
    prev = [-1]  * (m + 1)

    for t in range(1, m + 1):
        best_val, arg = -float("inf"), -1
        for i in range(1, t + 1):
            g0   = C0[i-1][t-1]
            val  = dp[i-1] + (g0 if g0 > EPS_COMPARE else 0.0)
            better = (val > best_val + EPS_COMPARE) or \
                     (abs(val - best_val) <= EPS_COMPARE and (i - 1) > arg)
            if better:
                best_val, arg = val, i - 1
        dp[t], prev[t] = best_val, arg

    q_star: List[List[Any]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star


# =============================================================================
# Conservative (no-false-positives) Alg.2 for bins (no Alg.1 edges)
# =============================================================================

def _bin_boundary_supported_interval_sufficient_envelope(
    Gi: List[Any], Gj: List[Any],
    *, idx: Dict[Any, int], k: int, prior: PriorSpec, biases: Dict[Any, float]
) -> bool:
    """
    Envelope-safe check for boundary Gi|Gj (Gi above Gj, domain is DESC).
    Uses worst-case ambiguous interval [theta_high(Gj), theta_low(Gi)) so there are NO false positives.
    """
    # Domain ranks (DESC) are contiguous within each bin since we only merge adjacent bins.
    i_start = min(idx[x] for x in Gi)  # 0-based index in domain
    i_end   = max(idx[x] for x in Gi)
    j_start = min(idx[x] for x in Gj)
    j_end   = max(idx[x] for x in Gj)

    # Convert to DESC ranks r_desc = index+1
    r_i_low  = i_end   + 1  # lowest theta in Gi
    r_j_high = j_start + 1  # highest theta in Gj

    theta_i_low  = prior.rank_expectation(k, r_i_low)
    theta_j_high = prior.rank_expectation(k, r_j_high)

    # If the envelope collapses or flips, nothing ambiguous remains
    if theta_i_low <= theta_j_high + EPS_COMPARE:
        return True

    Bi_min = min(biases[x] for x in Gi); Bi_max = max(biases[x] for x in Gi)
    Bj_min = min(biases[x] for x in Gj); Bj_max = max(biases[x] for x in Gj)

    # V-safe: all of Gj outside the interval
    vsafe = (Bj_max < theta_j_high - EPS_COMPARE) or (Bj_min >= theta_i_low + EPS_COMPARE)
    # U-safe: all of Gi inside the interval
    usafe = (Bi_min >= theta_j_high + EPS_COMPARE) and (Bi_max < theta_i_low - EPS_COMPARE)

    return vsafe or usafe


def algorithm_2_build_qbase_conservative_bins(
    q_bins: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float]
) -> List[List[Any]]:
    """
    Conservative Alg.2 operating on bins:
    - Uses the envelope-safe interval test above (no false positives).
    - Repeats merging unsupported adjacent bins until fixed point.
    """
    q_cur = [g[:] for g in q_bins]
    # map value -> 0-based index in DESC domain
    idx = {v: i for i, v in enumerate(domain)}
    k = len(domain)

    while True:
        merged = False
        for i in range(len(q_cur) - 1):
            Gi, Gj = q_cur[i], q_cur[i + 1]
            supported = _bin_boundary_supported_interval_sufficient_envelope(
                Gi, Gj, idx=idx, k=k, prior=prior, biases=biases
            )

            if not supported:
                # Conservative merge
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
        if not merged:
            return q_cur


# =============================================================================
# Bias helpers
# =============================================================================

def make_random_multilevel_bias(
    domain_values: List[Any],
    levels: Tuple[float, ...] = (0.8, 0.6, 0.3, 0.0),
    probs:  Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    seed: Optional[int] = 123,
) -> Dict[Any, float]:
    """
    Tuple-safe random multilevel bias: assigns each distinct value a level.
    Deterministic given seed and domain order.
    """
    rng = np.random.default_rng(seed)
    dom = list(domain_values)
    P = np.array(probs, dtype=float); P = P / P.sum()
    labels = rng.choice(len(levels), size=len(dom), p=P)
    return {dom[i]: float(levels[int(lbl)]) for i, lbl in enumerate(labels)}


# =============================================================================
# Binning helpers + experiment
# =============================================================================

def make_contiguous_bins(domain: List[Any], bins_count: int) -> List[List[Any]]:
    """
    Split the DESC-ordered domain into 'bins_count' contiguous slices.
    """
    n = len(domain)
    if bins_count <= 1:
        return [domain[:]]
    bins: List[List[Any]] = []
    for b in range(bins_count):
        start = (b * n) // bins_count
        end   = ((b + 1) * n) // bins_count
        if start < end:
            bins.append(domain[start:end])
    return bins

import pandas as pd
def load_amazon_df():
    candidate = "/Users/aryal/Desktop/Querying-COI/data/real/amazon_products.csv"
    if os.path.exists(candidate):
        try:
            df = pd.read_csv(candidate)
            #make price int
            print("Unique prices:", df['price'].nunique())
            df['price'] = df['price'].fillna(0)
            print("Unique prices:", df['price'].nunique())
            distinct_domain = sorted(set(df['price'].tolist()), reverse=True)
            return distinct_domain
        except Exception:
            pass
    #Synthetic fallback
    rng = np.random.default_rng(123)
    n = 5000
    df = pd.DataFrame({
        "age": rng.integers(18, 80, size=n),
        "education_num": rng.integers(1, 17, size=n),
        "sex": rng.choice(["Male", "Female"], size=n, p=[0.51, 0.49])
    })
    return df, False

def run_binwidth_sweep_experiment(
    domain_values: List[Any],
    *,
    bin_counts: List[int],
    bias_levels: Tuple[float, ...] = (0.8, 0.6, 0.3, 0.0),
    bias_probs:  Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    bias_seed: int = 123,
    csv_path: str = "./bin_sweep_results.csv",
    print_progress: bool = True,
):
    """
    Writes a CSV with rows for:
      - full:   q_base (Alg.2 original) + q_star (Alg.4)
      - binned: conservative Alg.2 + q_star (Alg.4), for each bins_count.

    Columns:
      kind, n, bin_size, bins_count,
      time_alg2_s, time_alg4_s,
      q_base_groups, q_star_groups,
      utility_babble, utility_base, utility_star,
      kept_babble, kept_base, kept_star
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    domain= sorted(set(domain_values), reverse=True)  # DESC, distinct
    n = len(domain)
    prior = PriorSpec(kind="uniform")
    biases = make_random_multilevel_bias(domain, levels=bias_levels, probs=bias_probs, seed=bias_seed)

    # Prepare CSV
    want_header = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = _csv.writer(f, quoting=_csv.QUOTE_NONNUMERIC)
    if want_header:
        w.writerow([
            "kind","n","bin_size","bins_count",
            "time_alg2_s","time_alg4_s",
            "q_base_groups","q_star_groups",
            "utility_babble","utility_base","utility_star",
            "kept_babble","kept_base","kept_star"
        ]); f.flush()

    # Baseline utilities
    util_babble, kept_babble = evaluate_plan_utility([domain[:]], domain, prior, biases=biases)


    # Binned (conservative) sweep
    for B in bin_counts:
        bins = make_contiguous_bins(domain, B)
        tba = time.perf_counter()
        q_base_bin = algorithm_2_build_qbase_conservative_bins(bins, domain, prior, biases=biases)
        tbb = time.perf_counter()
        q_star_bin = algorithm_4_maximally_informative(q_base_bin, domain, prior, biases=biases)
        tbc = time.perf_counter()

        util_base_bin, kept_base_bin = evaluate_plan_utility(q_base_bin, domain, prior, biases=biases)
        util_star_bin, kept_star_bin = evaluate_plan_utility(q_star_bin, domain, prior, biases=biases)

        w.writerow([
            "binned", n, max(1, n // max(1, B)), B,
            (tbb - tba), (tbc - tbb),
            len(q_base_bin), len(q_star_bin),
            util_babble, util_base_bin, util_star_bin,
            kept_babble, kept_base_bin, kept_star_bin
        ]); f.flush()

        if print_progress:
            print(f"[bins={B:4d}] q_base={len(q_base_bin):3d}→q*={len(q_star_bin):3d} | "
                  f"U(babble)={util_babble:.6g}, U(base)={util_base_bin:.6g}, U(q*)={util_star_bin:.6g} | "
                  f"t2={tbb-tba:.4f}s t4={tbc-tbb:.4f}s")

    #     # Full (item-level) baseline via original Alg.2 + Alg.4
    # t2a = time.perf_counter()
    # q_base_full = algorithm_2_build_qbase(domain, domain, prior, biases=biases,
    #                                       boundary_policy="all", alpha=0.5)
    # t2b = time.perf_counter()
    # q_star_full = algorithm_4_maximally_informative(q_base_full, domain, prior, biases=biases)
    # t2c = time.perf_counter()
    #
    # util_base_full, kept_base_full = evaluate_plan_utility(q_base_full, domain, prior, biases=biases)
    # util_star_full, kept_star_full = evaluate_plan_utility(q_star_full, domain, prior, biases=biases)
    #
    # w.writerow([
    #     "full", n, 1, n,
    #     (t2b - t2a), (t2c - t2b),
    #     len(q_base_full), len(q_star_full),
    #     util_babble, util_base_full, util_star_full,
    #     kept_babble, kept_base_full, kept_star_full
    # ]);
    # f.flush()
    #
    # if print_progress:
    #     print(f"[full] n={n} | q_base={len(q_base_full)}→q*={len(q_star_full)} | "
    #           f"U(babble)={util_babble:.6g}, U(base)={util_base_full:.6g}, U(q*)={util_star_full:.6g}")

    f.close()
    print(f"CSV -> {csv_path}")


# =============================================================================
# Example CLI: quick synthetic run (single-attribute domain)
# =============================================================================
if __name__ == "__main__":
    # Synthetic 1D domain: values are irrelevant; only order matters.

    domain = load_amazon_df()
    n = len(domain)
    print(f"Domain size n={n}")
    # Sweep a range of bin counts (coarse → fine). Include the full case (n). #bins= 3rootn
    bin_counts = [n/10000, n/5000, n/2000, n/1000, n/500]
    bin_counts = [max(1, int(b)) for b in bin_counts]

    run_binwidth_sweep_experiment(
        domain_values=domain,
        bin_counts=bin_counts,
        bias_levels=(0.8, 0.6, 0.3, 0),
        bias_probs=(0.25, 0.25, 0.25, 0.25),
        bias_seed=123,
        csv_path="./bin_sweep_results.csv",
        print_progress=True,
    )
