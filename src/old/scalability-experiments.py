# -*- coding: utf-8 -*-
"""
COI Scalability Pipeline (final, with MI monotonicity fix)
==========================================================

What's included:
- Priors, bias, best-response (threshold), and utilities.
- Alg.1 (true credibility), Alg.2 (reachability merges), Alg.4 (DP for maximally informative).
- DP fixed to use *marginal run gain vs base groups* and *forbid zero/negative-gain merges*,
  guaranteeing util_theta(q★) >= util_theta(q_base) under the reported metric.
- Calibrated monotone bias (top -> lower threshold, bottom -> higher) so ordering matters.
- 1D suite: full + grouped-start + bins-optimize→values-evaluate (collapsed removed to simplify).
- 2D suite: full + grouped-start + collapsed + cells-optimize→pairs-evaluate (you can disable modes).
- 5-minute per-experiment time budget (status="timeout" if exceeded).
- Keep signatures to confirm the receiver’s decisions actually change.

Run:
    python coi_scalability_pipeline.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import math
import time
from time import perf_counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Globals / tolerances
# =============================================================================
EPS_COMPARE = 1e-5
EPS_ORDER   = 1e-9
DEFAULT_SEED = 123


# =============================================================================
# Priors
# =============================================================================
@dataclass
class PriorSpec:
    kind: str = "uniform"
    a: float = 1.0
    b: float = 1.0
    lam: float = 1.0
    p: float = 1.0
    custom: Optional[callable] = None

    def rank_expectation(self, k: int, r_desc: int) -> float:
        if k <= 0 or r_desc <= 0 or r_desc > k:
            return 0.0

        if self.kind == "uniform":
            j = k - r_desc + 1  # ascending index
            return j / (k + 1.0)

        if self.kind == "beta":
            j = k - r_desc + 1
            den = k + self.a + self.b - 1.0
            if den <= 0: return 0.0
            return float(min(max((j + self.a - 1.0) / den, 0.0), 1.0))

        if self.kind == "exp_kernel":
            ws = [math.exp(-self.lam * (rr - 1)) for rr in range(1, k + 1)]
            wmax, wmin = max(ws), min(ws)
            if wmax == wmin: return 1.0
            return (ws[r_desc - 1] - wmin) / (wmax - wmin + EPS_COMPARE)

        if self.kind == "power_kernel":
            ws = [1.0 / (rr ** self.p) for rr in range(1, k + 1)]
            wmax, wmin = max(ws), min(ws)
            if wmax == wmin: return 1.0
            return (ws[r_desc - 1] - wmin) / (wmax - wmin + EPS_COMPARE)

        if self.kind == "custom" and self.custom:
            v = float(self.custom(k, r_desc))
            return float(min(max(v, 0.0), 1.0))

        # default
        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec) -> Dict[Any, float]:
    """θ under partition q: block-average of rank expectations."""
    k = len(domain)
    post: Dict[Any, float] = {}
    r = 1
    for g in q_groups:
        n = len(g)
        if n == 0: continue
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
# Bias
# =============================================================================
@dataclass
class Bias1D:
    kind: str
    degree: float = 1.0
    base: float = 0.0
    threshold: float = 0.0
    lo: float = 0.0
    hi: float = 1.0
    height: float = 1.0
    mu: float = 0.5
    sigma: float = 0.2
    ksig: float = 10.0
    center: float = 0.5
    knots_t: Optional[List[float]] = None
    knots_y: Optional[List[float]] = None
    custom: Optional[callable] = None

    @staticmethod
    def _norm(x: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    def bias_for_value(self, x: Any, domain_info: Dict[str, float]) -> float:
        if self.kind == "custom" and self.custom:
            b = float(self.custom(x, domain_info))
            return float(min(max(b, 0.0), 1.0))

        xv = float(x if not isinstance(x, tuple) else x[0])
        lo = float(domain_info["min"]); hi = float(domain_info["max"])
        t = self._norm(xv, lo, hi)

        if self.kind == "constant":
            b = self.base
        elif self.kind == "linear_high":
            b = self.degree * t
        elif self.kind == "linear_low":
            b = self.degree * (1.0 - t)
        elif self.kind == "step_value":
            b = self.degree if xv >= self.threshold else 0.0
        elif self.kind == "window":
            b = self.height if (xv >= self.lo and xv <= self.hi) else 0.0
        elif self.kind == "gaussian":
            b = self.degree * math.exp(- (xv - self.mu) ** 2 / (2.0 * (self.sigma ** 2) + EPS_COMPARE))
        elif self.kind == "sigmoid":
            b = self.degree * (1.0 / (1.0 + math.exp(-self.ksig * (t - self.center))))
        elif self.kind == "piecewise":
            if not self.knots_t or not self.knots_y or len(self.knots_t) != len(self.knots_y):
                b = self.base
            else:
                xs, ys = self.knots_t, self.knots_y
                if t <= xs[0]: b = ys[0]
                elif t >= xs[-1]: b = ys[-1]
                else:
                    i = np.searchsorted(xs, t) - 1
                    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
                    w = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
                    b = (1 - w) * y0 + w * y1
        else:
            b = self.base

        return float(min(max(b, 0.0), 1.0))


def _canon_key(x: Any) -> Any:
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, tuple): return tuple(_canon_key(t) for t in x)
    return x

def biases_from_bias_obj(domain: List[Any], bias_obj: Any) -> Dict[Any, float]:
    if not domain: return {}
    def _scalar(x): return float(x if not isinstance(x, tuple) else x[0])
    info = {"min": min(_scalar(v) for v in domain), "max": max(_scalar(v) for v in domain)}
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}

def random_levels_by_fraction(units: List[Any], levels: List[float], fracs: List[float], seed: Optional[int] = DEFAULT_SEED) -> Dict[Any, float]:
    if len(levels) != len(fracs): raise ValueError("levels and fracs must have same length")
    if not units: return {}
    rng = np.random.default_rng(seed)
    fr = np.array(fracs, dtype=float); fr = fr / (fr.sum() or 1.0)
    n = len(units); counts = (fr * n).astype(int)
    while counts.sum() < n: counts[np.argmax(fr)] += 1
    while counts.sum() > n: counts[np.argmax(counts)] -= 1
    units_shuf = units[:]; rng.shuffle(units_shuf)
    out: Dict[Any, float] = {}
    off = 0
    for lvl, c in zip(levels, counts.tolist()):
        for u in units_shuf[off:off+c]:
            out[_canon_key(u)] = float(max(0.0, min(1.0, lvl)))
        off += c
    return out

def bias_obj_from_map(bmap: Dict[Any, float]) -> Bias1D:
    bmap_c = {_canon_key(k): float(v) for k, v in bmap.items()}
    def f(x: Any, _info: Dict[str, float]) -> float:
        return float(bmap_c.get(_canon_key(x), 0.0))
    return Bias1D(kind='custom', custom=f)

def calibrated_rank_monotone_bias(domain_in_order: List[Any], *, cross_low: float = 0.40, cross_high: float = 0.60, noise_sd: float = 0.02, seed: int = DEFAULT_SEED) -> Dict[Any, float]:
    """Bias increases from top→bottom; crosses mid band so ordering matters."""
    k = len(domain_in_order)
    if k == 0: return {}
    rng = np.random.default_rng(seed)
    out = {}
    for i, v in enumerate(domain_in_order):
        t = i / max(1, k - 1)  # 0..1
        base = cross_low + t * (cross_high - cross_low)
        b = float(base + (rng.normal(0.0, noise_sd) if noise_sd > 0 else 0.0))
        out[_canon_key(v)] = float(min(max(b, 0.0), 1.0))
    return out


# =============================================================================
# Receiver + utilities
# =============================================================================
def _best_response_from_theta_threshold(theta: Dict[Any, float], biases: Dict[Any, float]) -> Dict[Any, int]:
    return {v: 1 if (theta.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0 for v in theta.keys()}

def system_best_response(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, biases: Dict[Any, float]) -> Dict[Any, int]:
    theta = compute_expected_posteriors(q_groups, domain, prior)
    return _best_response_from_theta_threshold(theta, biases)

def expected_utility_threshold_count_kept(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, biases: Dict[Any, float]) -> int:
    beta = system_best_response(q_groups, domain, prior, biases)
    return int(sum(int(beta.get(v, 0)) for v in domain))

def expected_utility_threshold_theta_weighted(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, biases: Dict[Any, float], *, theta_under: str = "singletons") -> float:
    """
    θ-weighted utility: ∑_v θ[v] · keep[v].
    IMPORTANT: default is 'singletons' (matches Alg.4 DP objective so q★ ≥ q_base).
    """
    if theta_under not in {"q", "singletons"}:
        raise ValueError("theta_under must be 'q' or 'singletons'")
    theta = compute_expected_posteriors(q_groups if theta_under == "q" else [[v] for v in domain], domain, prior)
    beta  = system_best_response(q_groups, domain, prior, biases)
    return float(sum(theta.get(v, 0.0) * float(beta.get(v, 0)) for v in domain))

def _exposure_weights(order_list: List[Any]) -> Dict[Any, float]:
    n = len(order_list)
    if n == 0: return {}
    raw = [1.0 / (r + 1) for r in range(n)]
    Z = sum(raw) or 1.0
    return {order_list[r]: raw[r] / Z for r in range(n)}

def user_utility_from_response(theta: Dict[Any, float], response: Dict[Any, float | int], *, order_list: Optional[List[Any]] = None, eps_order: float = 0.0) -> float:
    base = sum(theta.get(v, 0.0) * float(response.get(v, 0.0)) for v in theta.keys())
    if eps_order > 0.0 and order_list is not None:
        w = _exposure_weights(order_list)
        bonus = sum(w.get(v, 0.0) * theta.get(v, 0.0) * float(response.get(v, 0.0)) for v in order_list)
        return base + eps_order * bonus
    return base


# =============================================================================
# Helpers
# =============================================================================
def dedupe_and_sort_desc(values: List[Any]) -> List[Any]:
    return sorted(set(values), reverse=True)

def flatten_in_order(q_groups: List[List[Any]]) -> List[Any]:
    return [x for g in q_groups for x in g]

def op_pairs_strict(q_groups: List[List[Any]]) -> List[Tuple[Any, Any]]:
    pairs: List[Tuple[Any, Any]] = []
    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def swap_single_pair(q_groups: List[List[Any]], u: Any, v: Any) -> List[List[Any]]:
    return [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]

def keep_signature(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, biases: Dict[Any, float]) -> str:
    beta = system_best_response(q_groups, domain, prior, biases)
    vec = tuple(int(beta.get(v, 0)) for v in domain)
    return hex(abs(hash(vec)) % (1 << 32))[2:]


# =============================================================================
# Alg.1 (true credibility, threshold)
# =============================================================================
def algorithm_1_credibility_detection(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *, biases: Dict[Any, float], eps_order: float = 0.0, cred_rule: str = "strict_opposite") -> set[tuple[Any, Any]]:
    C: set[tuple[Any, Any]] = set()

    def _decide(a: float, b: float) -> Optional[str]:
        if a > b + EPS_COMPARE: return "q"
        if b > a + EPS_COMPARE: return "swap"
        return None

    theta_plus = compute_expected_posteriors(q_groups, domain, prior)
    beta_q     = _best_response_from_theta_threshold(theta_plus, biases)
    order_q    = flatten_in_order(q_groups)

    for (u, v) in op_pairs_strict(q_groups):
        q_swap    = swap_single_pair(q_groups, u, v)
        theta_min = compute_expected_posteriors(q_swap, domain, prior)
        beta_s    = _best_response_from_theta_threshold(theta_min, biases)
        order_s   = flatten_in_order(q_swap)

        up_q = user_utility_from_response(theta_plus, beta_q, order_list=order_q, eps_order=eps_order)
        up_s = user_utility_from_response(theta_plus, beta_s, order_list=order_s, eps_order=eps_order)

        um_q = user_utility_from_response(theta_min,  beta_q, order_list=order_q, eps_order=eps_order)
        um_s = user_utility_from_response(theta_min,  beta_s, order_list=order_s, eps_order=eps_order)

        cplus  = _decide(up_q, up_s)
        cminus = _decide(um_q, um_s)
        if cred_rule == "strict_opposite":
            if cplus is not None and cminus is not None and cplus != cminus:
                C.add((u, v))
        else:
            both_q = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
            both_s = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
            if not (both_q or both_s):
                C.add((u, v))

    return C


# =============================================================================
# Alg.2 (reachability merges)
# =============================================================================
def _build_adj(items: List[Any], edges: set[tuple[Any, Any]]) -> Dict[Any, List[Any]]:
    adj = {x: [] for x in items}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)
    return adj

def _reachable(start: Any, adj: Dict[Any, List[Any]]) -> set[Any]:
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for w in adj.get(u, []):
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return seen

def _merge_once_if_not_supported(q_cur: List[List[Any]], domain: List[Any], C: set[tuple[Any, Any]]) -> Tuple[List[List[Any]], bool]:
    adj = _build_adj(domain, C)
    i = 0
    while i < len(q_cur) - 1:
        Gi, Gj = q_cur[i], q_cur[i + 1]
        if not all(v in _reachable(u, adj) for u in Gi for v in Gj):
            return q_cur[:i] + [Gi + Gj] + q_cur[i + 2:], True
        i += 1
    return q_cur, False

def algorithm_2_build_qbase(initial_order: List[Any], domain: List[Any], prior: PriorSpec, *, biases: Dict[Any, float], eps_order: float = 0.0, cred_rule: str = "strict_opposite", timing: Optional[Dict[str, float]] = None, time_budget_sec: Optional[float] = None) -> List[List[Any]]:
    if timing is None: timing = {}
    time_cred_accum = 0.0
    t_start = time.perf_counter()

    q_cur: List[List[Any]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        if time_budget_sec is not None and (time.perf_counter() - t_start) > time_budget_sec:
            break
        t0 = time.perf_counter()
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases, eps_order=eps_order, cred_rule=cred_rule)
        time_cred_accum += (time.perf_counter() - t0)
        timing["time_cred"] = time_cred_accum
        q_next, merged = _merge_once_if_not_supported(q_cur, domain, C)
        if not merged:
            break
        q_cur = q_next
    return q_cur

def algorithm_2_build_qbase_from_groups(user_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *, biases: Dict[Any, float], eps_order: float = 0.0, cred_rule: str = "strict_opposite", timing: Optional[Dict[str, float]] = None, time_budget_sec: Optional[float] = None) -> List[List[Any]]:
    if timing is None: timing = {}
    time_cred_accum = 0.0
    t_start = time.perf_counter()
    q_cur = [g[:] for g in user_groups]

    for _ in range(len(domain) + 1):
        if time_budget_sec is not None and (time.perf_counter() - t_start) > time_budget_sec:
            break
        t0 = time.perf_counter()
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases, eps_order=eps_order, cred_rule=cred_rule)
        time_cred_accum += (time.perf_counter() - t0)
        timing["time_cred"] = time_cred_accum
        q_next, merged = _merge_once_if_not_supported(q_cur, domain, C)
        if not merged:
            break
        q_cur = q_next
    return q_cur


# =============================================================================
# Alg.4 (DP) — FIXED: maximize sum of strictly positive marginal gains vs q_base
# =============================================================================
def algorithm_4_maximally_informative(q_base: List[List[Any]], domain: List[Any], prior: PriorSpec, *, biases: Dict[Any, float], eps_order_for_tiebreak: float = 0.0) -> List[List[Any]]:
    """
    We optimize ∑ θ_single[v]*keep[v] (plus tiny tie-break bonus) and ensure:
        util_theta(q★) >= util_theta(q_base)  (with θ_single)
    by using DELTA gains vs base groups and forbidding non-positive merges.
    """
    m = len(q_base)
    if m == 0:
        return []

    # θ on singletons (fixed across candidates)
    theta_single = compute_expected_posteriors([[v] for v in domain], domain, prior)

    # Base full best-response
    beta_base = system_best_response(q_base, domain, prior, biases)

    # Per-base-group baseline contributions (for gain deltas)
    base_gain0   = [0.0] * m
    base_gain_eps = [0.0] * m
    for t, Gt in enumerate(q_base):
        resp_t = {x: beta_base.get(x, 0) for x in Gt}
        base_gain0[t] = user_utility_from_response(theta_single, resp_t, order_list=None, eps_order=0.0)
        base_gain_eps[t] = user_utility_from_response(theta_single, resp_t, order_list=Gt, eps_order=eps_order_for_tiebreak)

    # Marginal gain matrices
    C0 = [[0.0]*m for _ in range(m)]
    Ceps = [[0.0]*m for _ in range(m)]

    for i in range(m):
        run_vals: List[Any] = []
        for j in range(i, m):
            run_vals.extend(q_base[j])

            # Partition with i..j merged
            temp_q = q_base[:i] + [run_vals] + q_base[j+1:]
            beta_run = system_best_response(temp_q, domain, prior, biases)
            resp_run = {x: beta_run.get(x, 0) for x in run_vals}

            u0_run   = user_utility_from_response(theta_single, resp_run, order_list=None, eps_order=0.0)
            ueps_run = user_utility_from_response(theta_single, resp_run, order_list=run_vals, eps_order=eps_order_for_tiebreak)

            base_line0  = sum(base_gain0[t]   for t in range(i, j+1))
            base_lineep = sum(base_gain_eps[t] for t in range(i, j+1))

            C0[i][j]   = u0_run   - base_line0    # strictly positive ⇒ merge is useful
            Ceps[i][j] = ueps_run - base_lineep   # tie-break delta

    # If no positive gains anywhere, keep q_base
    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    # DP with "no-merge" option per step
    dp  = [0.0] * (m + 1)
    tie = [0.0] * (m + 1)
    prev = [-1] * (m + 1)
    for t in range(1, m + 1):
        # candidate: don't merge at the end (keep group t-1 as-is)
        best_val = dp[t-1]
        best_tie = tie[t-1]
        arg = t-1

        # candidates: merge i..t-1 if it yields strictly positive gain
        for i in range(1, t + 1):
            g0   = C0[i-1][t-1]
            geps = Ceps[i-1][t-1]
            if g0 > EPS_COMPARE:
                val2 = dp[i-1] + g0
                tie2 = tie[i-1] + geps
                better = (val2 > best_val + EPS_COMPARE) or \
                         (abs(val2 - best_val) <= EPS_COMPARE and (tie2 > best_tie + EPS_COMPARE)) or \
                         (abs(val2 - best_val) <= EPS_COMPARE and abs(tie2 - best_tie) <= EPS_COMPARE and (i - 1) > arg)
                if better:
                    best_val, best_tie, arg = val2, tie2, i - 1

        dp[t], tie[t], prev[t] = best_val, best_tie, arg

    # Reconstruct q★
    q_star: List[List[Any]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star


# =============================================================================
# I/O + discretization
# =============================================================================
def load_numeric(df_path: str, col: str) -> pd.Series:
    df = pd.read_csv(df_path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in {df_path}")
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if s.empty:
        raise ValueError(f"Column '{col}' has no numeric values in {df_path}")
    return s

def load_two_numeric(df_path: str, col1: str, col2: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(df_path)
    for c in (col1, col2):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in {df_path}")
    a = pd.to_numeric(df[col1], errors='coerce')
    b = pd.to_numeric(df[col2], errors='coerce')
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if a.empty:
        raise ValueError(f"No overlapping numeric rows for '{col1}','{col2}' in {df_path}")
    return a, b

def quantile_bins_from_series(s: pd.Series, nbins: int) -> List[Tuple[float, Optional[float]]]:
    if nbins <= 0:
        raise ValueError("nbins must be > 0")
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.unique(np.quantile(s.values, qs))
    if len(edges) < 2:
        lo = float(np.min(s.values))
        return [(lo, None)]
    bins: List[Tuple[float, Optional[float]]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if hi == lo: continue
        bins.append((lo, hi))
    if not bins:
        lo = float(np.min(s.values)); bins = [(lo, None)]
    else:
        lo_last, _ = bins[-1]; bins[-1] = (lo_last, None)
    return bins

def groups_from_bins_desc(domain_desc: List[Any], bins: List[Tuple[float, Optional[float]]]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for lo, hi in bins:
        S = set(v for v in domain_desc if (float(v) >= lo) and (hi is None or float(v) < hi))
        grp = [v for v in domain_desc if v in S]
        if grp:
            out.append(grp)
    return out

def bucket_partition_from_bins(domain_desc: List[float], bins: List[Tuple[float, Optional[float]]]) -> tuple[List[List[float]], List[int], Dict[float, int]]:
    groups = groups_from_bins_desc(domain_desc, bins)
    bucket_ids = list(range(len(groups)))
    v2b: Dict[float, int] = {}
    for bi, grp in enumerate(groups):
        for v in grp:
            v2b[v] = bi
    return groups, bucket_ids, v2b

def grid_cells_from_bins_desc_2d(domain_pairs_desc: List[Tuple[float, float]], bins_a: List[Tuple[float, Optional[float]]], bins_b: List[Tuple[float, Optional[float]]]) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    def idx_in_bins(x: float, bins: List[Tuple[float, Optional[float]]]) -> Optional[int]:
        xv = float(x)
        for i, (lo, hi) in enumerate(bins):
            if xv >= lo and (hi is None or xv < hi):
                return i
        return None
    cell2pairs: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for (a, b) in domain_pairs_desc:
        ia = idx_in_bins(a, bins_a); ib = idx_in_bins(b, bins_b)
        if ia is None or ib is None: continue
        cell2pairs.setdefault((ia, ib), []).append((a, b))
    for k in list(cell2pairs.keys()):
        pairs = cell2pairs[k]; present = set(pairs)
        cell2pairs[k] = [p for p in domain_pairs_desc if p in present]
    return cell2pairs


# =============================================================================
# Expansion
# =============================================================================
def expand_groups_from_idmap(q_groups_ids: List[List[int]], id2members_desc: Dict[int, List[Any]]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for block in q_groups_ids:
        big = []
        for cid in block:
            big.extend(id2members_desc[cid])
        out.append(big)
    return out


# =============================================================================
# Tidy rows
# =============================================================================
@dataclass
class RunOutcome:
    mode: str
    bias_kind: str
    label: str
    domain_size: int
    nbins: float
    nbins_age: float
    nbins_priors: float
    bucket_count: float
    grid_cells: float
    groups: int
    time_baseline: float
    time_qbase: float
    time_cred: float
    time_mi: float
    util_count: int
    util_theta: float
    status: str
    keep_sig: str

def _pack(rows: List[RunOutcome]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    cols = ["mode","bias_kind","label","domain_size","nbins","nbins_age","nbins_priors",
            "bucket_count","grid_cells","groups","time_baseline","time_qbase","time_cred","time_mi",
            "util_count","util_theta","status","keep_sig"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols]

def _timeout(now: float, start: float, budget: Optional[float]) -> bool:
    return (budget is not None) and ((now - start) > budget)


# =============================================================================
# 1D runners (full, grouped-start, bins-opt-eval-values)  ← collapsed removed
# =============================================================================
def run_1d_full(s_age: pd.Series, *, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = None) -> pd.DataFrame:
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')
    domain = dedupe_and_sort_desc(s_age.unique().tolist())
    domain_size = len(domain)

    # bias
    bias_map = calibrated_rank_monotone_bias(domain, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed) \
               if use_calibrated_bias else random_levels_by_fraction(domain, [0.9,0.8,0.6,0.4,0.2],[0.1,0.1,0.2,0.3,0.3],seed)
    biases = biases_from_bias_obj(domain, bias_obj_from_map(bias_map))

    rows: List[RunOutcome] = []

    # baseline (single block)
    tb0 = perf_counter()
    q_babble = [domain[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble, domain, prior, biases)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble, domain, prior, biases, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble, domain, prior, biases)
    rows.append(RunOutcome("full","per_value","baseline",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("full","per_value","q_base",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    # Alg.2 (true)
    timing = {}
    ta0 = perf_counter()
    q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, eps_order=1e-9, cred_rule="strict_opposite",
                                     timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()
    sig_qb = keep_signature(q_base, domain, prior, biases)

    # Alg.4 (DP with delta-gains)
    ts0 = perf_counter()
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()
    sig_qs = keep_signature(q_star, domain, prior, biases)

    # utilities (θ_singletons to match DP objective => q★ ≥ q_base)
    u_cnt_base = expected_utility_threshold_count_kept(q_base, domain, prior, biases)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base, domain, prior, biases, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star, domain, prior, biases)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star, domain, prior, biases, theta_under="singletons")

    rows += [
        RunOutcome("full","per_value","q_base",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,len(q_base),
                   0.0, ta1-ta0, timing.get("time_cred",0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("full","per_value","q_star",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,len(q_star),
                   0.0, ta1-ta0, timing.get("time_cred",0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)

def run_1d_grouped_start(s_age: pd.Series, *, nbins: int, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = None) -> pd.DataFrame:
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')
    domain = dedupe_and_sort_desc(s_age.unique().tolist())
    domain_size = len(domain)

    # bins (just a starting partial order)
    bins = quantile_bins_from_series(s_age, nbins)
    q_user = groups_from_bins_desc(domain, bins)

    # bias
    bias_map = calibrated_rank_monotone_bias(domain, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed) \
               if use_calibrated_bias else random_levels_by_fraction(domain, [0.9,0.8,0.6,0.4,0.2],[0.1,0.1,0.2,0.3,0.3],seed)
    biases = biases_from_bias_obj(domain, bias_obj_from_map(bias_map))

    rows: List[RunOutcome] = []

    # baseline
    tb0 = perf_counter()
    q_babble = [domain[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble, domain, prior, biases)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble, domain, prior, biases, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble, domain, prior, biases)
    rows.append(RunOutcome("grouped-start","per_value","baseline",domain_size,nbins,np.nan,np.nan,np.nan,np.nan,1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("grouped-start","per_value","q_base",domain_size,nbins,np.nan,np.nan,np.nan,np.nan,1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    # Alg.2 starting from user ties
    timing = {}
    ta0 = perf_counter()
    q_base = algorithm_2_build_qbase_from_groups(q_user, domain, prior, biases=biases, eps_order=1e-9, cred_rule="strict_opposite",
                                                 timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()
    sig_qb = keep_signature(q_base, domain, prior, biases)

    # Alg.4
    ts0 = perf_counter()
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()
    sig_qs = keep_signature(q_star, domain, prior, biases)

    u_cnt_base = expected_utility_threshold_count_kept(q_base, domain, prior, biases)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base, domain, prior, biases, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star, domain, prior, biases)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star, domain, prior, biases, theta_under="singletons")

    rows += [
        RunOutcome("grouped-start","per_value","q_base",domain_size,nbins,np.nan,np.nan,np.nan,np.nan,len(q_base),
                   0.0, ta1-ta0, timing.get("time_cred",0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("grouped-start","per_value","q_star",domain_size,nbins,np.nan,np.nan,np.nan,np.nan,len(q_star),
                   0.0, ta1-ta0, timing.get("time_cred",0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)

def run_1d_bins_optimize_value_eval(s_age: pd.Series, *, nbins: int, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = None) -> pd.DataFrame:
    """
    Build on bin domain (fast), expand to values for eval with per-value bias.
    """
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')
    domain_vals = dedupe_and_sort_desc(s_age.unique().tolist())
    domain_size = len(domain_vals)

    bins = quantile_bins_from_series(s_age, nbins)
    groups = groups_from_bins_desc(domain_vals, bins)
    B = len(groups)
    domain_bins = list(range(B))
    binid2vals = {i: groups[i] for i in range(B)}

    if use_calibrated_bias:
        unit2bias_bins = calibrated_rank_monotone_bias(domain_bins, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed)
        unit2bias_vals = calibrated_rank_monotone_bias(domain_vals, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed+1)
    else:
        unit2bias_bins = random_levels_by_fraction(domain_bins, [0.9,0.8,0.6,0.4,0.2],[0.1,0.1,0.2,0.3,0.3],seed)
        unit2bias_vals = random_levels_by_fraction(domain_vals, [0.9,0.8,0.6,0.4,0.2],[0.1,0.1,0.2,0.3,0.3],seed+1)

    biases_bins = biases_from_bias_obj(domain_bins, bias_obj_from_map(unit2bias_bins))
    biases_vals = biases_from_bias_obj(domain_vals, bias_obj_from_map(unit2bias_vals))

    rows: List[RunOutcome] = []

    # baseline on values
    tb0 = perf_counter()
    q_babble_vals = [domain_vals[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble_vals, domain_vals, prior, biases_vals)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble_vals, domain_vals, prior, biases_vals, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble_vals, domain_vals, prior, biases_vals)
    rows.append(RunOutcome("bins-opt-eval-values","per_value","baseline",
                           domain_size, nbins, np.nan, np.nan, B, np.nan, 1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("bins-opt-eval-values","per_value","q_base",
                                        domain_size, nbins, np.nan, np.nan, B, np.nan, 1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    # build on bins
    timing = {}
    ta0 = perf_counter()
    q_base_bins = algorithm_2_build_qbase(domain_bins, domain_bins, prior, biases=biases_bins, eps_order=1e-9, cred_rule="strict_opposite",
                                          timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()

    ts0 = perf_counter()
    q_star_bins = algorithm_4_maximally_informative(q_base_bins, domain_bins, prior, biases=biases_bins, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()

    # expand + eval on values
    q_base_vals = expand_groups_from_idmap(q_base_bins, binid2vals)
    q_star_vals = expand_groups_from_idmap(q_star_bins, binid2vals)

    sig_qb = keep_signature(q_base_vals, domain_vals, prior, biases_vals)
    sig_qs = keep_signature(q_star_vals, domain_vals, prior, biases_vals)

    u_cnt_base = expected_utility_threshold_count_kept(q_base_vals, domain_vals, prior, biases_vals)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base_vals, domain_vals, prior, biases_vals, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star_vals, domain_vals, prior, biases_vals)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star_vals, domain_vals, prior, biases_vals, theta_under="singletons")

    rows += [
        RunOutcome("bins-opt-eval-values","per_value","q_base",
                   domain_size, nbins, np.nan, np.nan, B, np.nan, len(q_base_bins),
                   0.0, ta1-ta0, timing.get("time_cred", 0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("bins-opt-eval-values","per_value","q_star",
                   domain_size, nbins, np.nan, np.nan, B, np.nan, len(q_star_bins),
                   0.0, ta1-ta0, timing.get("time_cred", 0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)


# =============================================================================
# 2D runners (kept as before; you can disable modes if desired)
# =============================================================================
def run_2d_full(s_age: pd.Series, s_pr: pd.Series, *, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0) -> pd.DataFrame:
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')
    pairs = list(zip(s_age.values.tolist(), s_pr.values.tolist()))
    domain = sorted(set(pairs), key=lambda t:(t[0],t[1]), reverse=True)
    domain_size = len(domain)

    bias_map = calibrated_rank_monotone_bias(domain, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed) \
               if use_calibrated_bias else random_levels_by_fraction(domain, [0.95,0.8,0.5,0.2,0.0],[0.1,0.1,0.2,0.3,0.3],seed)
    biases = biases_from_bias_obj(domain, bias_obj_from_map(bias_map))

    rows: List[RunOutcome] = []

    tb0 = perf_counter()
    q_babble = [domain[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble, domain, prior, biases)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble, domain, prior, biases, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble, domain, prior, biases)
    rows.append(RunOutcome("full2d","per_value","baseline",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("full2d","per_value","q_base",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    timing = {}
    ta0 = perf_counter()
    q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, eps_order=1e-9, cred_rule="strict_opposite",
                                     timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()
    sig_qb = keep_signature(q_base, domain, prior, biases)

    ts0 = perf_counter()
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()
    sig_qs = keep_signature(q_star, domain, prior, biases)

    u_cnt_base = expected_utility_threshold_count_kept(q_base, domain, prior, biases)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base, domain, prior, biases, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star, domain, prior, biases)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star, domain, prior, biases, theta_under="singletons")

    rows += [
        RunOutcome("full2d","per_value","q_base",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,len(q_base),
                   0.0, ta1-ta0, timing.get('time_cred',0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("full2d","per_value","q_star",domain_size,np.nan,np.nan,np.nan,np.nan,np.nan,len(q_star),
                   0.0, ta1-ta0, timing.get('time_cred',0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)

def run_2d_grouped_start(s_age: pd.Series, s_pr: pd.Series, *, nbins_age: int, nbins_pr: int, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0) -> pd.DataFrame:
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')
    pairs = list(zip(s_age.values.tolist(), s_pr.values.tolist()))
    domain = sorted(set(pairs), key=lambda t:(t[0],t[1]), reverse=True)
    domain_size = len(domain)

    bias_map = calibrated_rank_monotone_bias(domain, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed) \
               if use_calibrated_bias else random_levels_by_fraction(domain, [0.95,0.8,0.5,0.2,0.0],[0.1,0.1,0.2,0.3,0.3],seed)
    biases = biases_from_bias_obj(domain, bias_obj_from_map(bias_map))

    rows: List[RunOutcome] = []

    tb0 = perf_counter()
    q_babble = [domain[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble, domain, prior, biases)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble, domain, prior, biases, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble, domain, prior, biases)
    rows.append(RunOutcome("grouped-start2d","per_value","baseline",domain_size,np.nan,nbins_age,nbins_pr,np.nan,np.nan,1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("grouped-start2d","per_value","q_base",domain_size,np.nan,nbins_age,nbins_pr,np.nan,np.nan,1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    timing = {}
    ta0 = perf_counter()
    q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, eps_order=1e-9, cred_rule="strict_opposite",
                                     timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()
    sig_qb = keep_signature(q_base, domain, prior, biases)

    ts0 = perf_counter()
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()
    sig_qs = keep_signature(q_star, domain, prior, biases)

    u_cnt_base = expected_utility_threshold_count_kept(q_base, domain, prior, biases)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base, domain, prior, biases, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star, domain, prior, biases)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star, domain, prior, biases, theta_under="singletons")

    rows += [
        RunOutcome("grouped-start2d","per_value","q_base",domain_size,np.nan,nbins_age,nbins_pr,np.nan,np.nan,len(q_base),
                   0.0, ta1-ta0, timing.get('time_cred',0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("grouped-start2d","per_value","q_star",domain_size,np.nan,nbins_age,nbins_pr,np.nan,np.nan,len(q_star),
                   0.0, ta1-ta0, timing.get('time_cred',0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)

def run_2d_collapsed(s_age: pd.Series, s_pr: pd.Series, *, nbins_age: int, nbins_pr: int, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0) -> pd.DataFrame:
    # unchanged from before (left in case you still want it for 2D).
    prior = PriorSpec(kind='uniform')
    pairs = list(zip(s_age.values.tolist(), s_pr.values.tolist()))
    domain_vals = sorted(set(pairs), key=lambda t:(t[0],t[1]), reverse=True)
    domain_size = len(domain_vals)

    bins_a = quantile_bins_from_series(s_age, nbins_age)
    bins_b = quantile_bins_from_series(s_pr,  nbins_pr)
    cell2pairs = grid_cells_from_bins_desc_2d(domain_vals, bins_a, bins_b)
    present_cells = sorted(cell2pairs.keys(), key=lambda ij:(ij[0],ij[1]), reverse=True)
    K = len(present_cells)
    domain_coarse = list(range(K))
    cellid2pairs = {cid: cell2pairs[ij] for cid, ij in enumerate(present_cells)}

    bias_map_c = calibrated_rank_monotone_bias(domain_coarse, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed) \
                 if use_calibrated_bias else random_levels_by_fraction(domain_coarse, [0.95,0.8,0.5,0.2,0.0],[0.1,0.1,0.2,0.3,0.3],seed)
    biases_c = biases_from_bias_obj(domain_coarse, bias_obj_from_map(bias_map_c))

    rows: List[RunOutcome] = []

    t0 = perf_counter()
    q_babble_c = [domain_coarse[:]]
    beta_b = system_best_response(q_babble_c, domain_coarse, prior, biases_c)
    theta_c = compute_expected_posteriors([[i] for i in domain_coarse], domain_coarse, prior)
    util_count_b = sum((1 if beta_b[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)
    util_theta_b = sum(theta_c[i] * (1 if beta_b[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)
    t1 = perf_counter()
    sig_b = keep_signature(q_babble_c, domain_coarse, prior, biases_c)
    rows.append(RunOutcome("collapsed2d","per_bucket2d","baseline",domain_size,np.nan,nbins_age,nbins_pr,np.nan,K,1,
                           t1-t0,0.0,0.0,0.0,int(util_count_b),float(util_theta_b),"ok",sig_b))

    timing = {}
    ta0 = perf_counter()
    q_base_c = algorithm_2_build_qbase(domain_coarse, domain_coarse, prior, biases=biases_c, eps_order=1e-9, cred_rule="strict_opposite",
                                       timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()
    sig_qb = keep_signature(q_base_c, domain_coarse, prior, biases_c)

    ts0 = perf_counter()
    q_star_c = algorithm_4_maximally_informative(q_base_c, domain_coarse, prior, biases=biases_c, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()
    sig_qs = keep_signature(q_star_c, domain_coarse, prior, biases_c)

    beta_base_c = system_best_response(q_base_c, domain_coarse, prior, biases_c)
    beta_star_c = system_best_response(q_star_c, domain_coarse, prior, biases_c)
    util_count_base = sum((1 if beta_base_c[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)
    util_theta_base = sum(theta_c[i] * (1 if beta_base_c[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)
    util_count_star = sum((1 if beta_star_c[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)
    util_theta_star = sum(theta_c[i] * (1 if beta_star_c[i] else 0) * len(cellid2pairs[i]) for i in domain_coarse)

    rows += [
        RunOutcome("collapsed2d","per_bucket2d","q_base",domain_size,np.nan,nbins_age,nbins_pr,np.nan,K,len(q_base_c),
                   0.0,ta1-ta0,timing.get('time_cred',0.0),0.0,int(util_count_base),float(util_theta_base),"ok",sig_qb),
        RunOutcome("collapsed2d","per_bucket2d","q_star",domain_size,np.nan,nbins_age,nbins_pr,np.nan,K,len(q_star_c),
                   0.0,ta1-ta0,timing.get('time_cred',0.0),ts1-ts0,int(util_count_star),float(util_theta_star),"ok",sig_qs),
    ]
    return _pack(rows)

def run_2d_cells_optimize_pair_eval(s_age: pd.Series, s_pr: pd.Series, *, nbins_age: int, nbins_pr: int, use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0) -> pd.DataFrame:
    t0_all = perf_counter()
    prior = PriorSpec(kind='uniform')

    pairs = list(zip(s_age.values.tolist(), s_pr.values.tolist()))
    domain_pairs = sorted(set(pairs), key=lambda t:(t[0],t[1]), reverse=True)
    domain_size  = len(domain_pairs)

    bins_a = quantile_bins_from_series(s_age, nbins_age)
    bins_b = quantile_bins_from_series(s_pr,  nbins_pr)
    cell2pairs = grid_cells_from_bins_desc_2d(domain_pairs, bins_a, bins_b)
    present_cells = sorted(cell2pairs.keys(), key=lambda ij:(ij[0],ij[1]), reverse=True)
    K = len(present_cells)
    domain_cells = list(range(K))
    cellid2pairs = {cid: cell2pairs[ij] for cid, ij in enumerate(present_cells)}

    if use_calibrated_bias:
        unit2bias_cells = calibrated_rank_monotone_bias(domain_cells, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed)
        unit2bias_pairs = calibrated_rank_monotone_bias(domain_pairs, cross_low=0.40, cross_high=0.60, noise_sd=0.02, seed=seed+1)
    else:
        unit2bias_cells = random_levels_by_fraction(domain_cells, [0.95,0.8,0.5,0.2,0.0],[0.1,0.1,0.2,0.3,0.3],seed)
        unit2bias_pairs = random_levels_by_fraction(domain_pairs,[0.95,0.8,0.5,0.2,0.0],[0.1,0.1,0.2,0.3,0.3],seed+1)
    biases_cells = biases_from_bias_obj(domain_cells, bias_obj_from_map(unit2bias_cells))
    biases_pairs = biases_from_bias_obj(domain_pairs, bias_obj_from_map(unit2bias_pairs))

    rows: List[RunOutcome] = []

    tb0 = perf_counter()
    q_babble_pairs = [domain_pairs[:]]
    u_cnt_b  = expected_utility_threshold_count_kept(q_babble_pairs, domain_pairs, prior, biases_pairs)
    u_tht_b  = expected_utility_threshold_theta_weighted(q_babble_pairs, domain_pairs, prior, biases_pairs, theta_under="singletons")
    tb1 = perf_counter()
    sig_b = keep_signature(q_babble_pairs, domain_pairs, prior, biases_pairs)
    rows.append(RunOutcome("cells-opt-eval-pairs","per_value","baseline",
                           domain_size, np.nan, nbins_age, nbins_pr, np.nan, K, 1,
                           tb1-tb0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "ok", sig_b))

    if _timeout(perf_counter(), t0_all, time_budget_sec):
        return _pack(rows + [RunOutcome("cells-opt-eval-pairs","per_value","q_base",
                                        domain_size, np.nan, nbins_age, nbins_pr, np.nan, K, 1,
                                        0.0, 0.0, 0.0, 0.0, u_cnt_b, u_tht_b, "timeout", sig_b)])

    timing = {}
    ta0 = perf_counter()
    q_base_cells = algorithm_2_build_qbase(domain_cells, domain_cells, prior, biases=biases_cells, eps_order=1e-9, cred_rule="strict_opposite",
                                           timing=timing, time_budget_sec=time_budget_sec)
    ta1 = perf_counter()

    ts0 = perf_counter()
    q_star_cells = algorithm_4_maximally_informative(q_base_cells, domain_cells, prior, biases=biases_cells, eps_order_for_tiebreak=1e-12)
    ts1 = perf_counter()

    q_base_pairs = expand_groups_from_idmap(q_base_cells, cellid2pairs)
    q_star_pairs = expand_groups_from_idmap(q_star_cells, cellid2pairs)

    sig_qb = keep_signature(q_base_pairs, domain_pairs, prior, biases_pairs)
    sig_qs = keep_signature(q_star_pairs, domain_pairs, prior, biases_pairs)

    u_cnt_base = expected_utility_threshold_count_kept(q_base_pairs, domain_pairs, prior, biases_pairs)
    u_tht_base = expected_utility_threshold_theta_weighted(q_base_pairs, domain_pairs, prior, biases_pairs, theta_under="singletons")
    u_cnt_star = expected_utility_threshold_count_kept(q_star_pairs, domain_pairs, prior, biases_pairs)
    u_tht_star = expected_utility_threshold_theta_weighted(q_star_pairs, domain_pairs, prior, biases_pairs, theta_under="singletons")

    rows += [
        RunOutcome("cells-opt-eval-pairs","per_value","q_base",
                   domain_size, np.nan, nbins_age, nbins_pr, np.nan, K, len(q_base_cells),
                   0.0, ta1-ta0, timing.get("time_cred", 0.0), 0.0, u_cnt_base, u_tht_base, "ok", sig_qb),
        RunOutcome("cells-opt-eval-pairs","per_value","q_star",
                   domain_size, np.nan, nbins_age, nbins_pr, np.nan, K, len(q_star_cells),
                   0.0, ta1-ta0, timing.get("time_cred", 0.0), ts1-ts0, u_cnt_star, u_tht_star, "ok", sig_qs),
    ]
    return _pack(rows)


# =============================================================================
# Batch suites
# =============================================================================
def run_scalability_suite_1d(dataset_path: str, age_col: str, *, discretized_bins: List[int] = [8, 16, 32, 64], use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0, out_csv: Optional[str] = None) -> pd.DataFrame:
    s_age = load_numeric(dataset_path, age_col)
    dfs: List[pd.DataFrame] = []
    print(f"[1D suite] rows={len(s_age)} col={age_col} from {dataset_path}")

    dfs.append(run_1d_full(s_age, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))

    for nb in discretized_bins:
        dfs.append(run_1d_grouped_start(s_age, nbins=nb, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))
        dfs.append(run_1d_bins_optimize_value_eval(s_age, nbins=nb, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))

    out = pd.concat(dfs, ignore_index=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"[suite_1d] wrote {out_csv}")
    return out

def run_scalability_suite_2d(dataset_path: str, age_col: str, priors_col: str, *, discretized_bins_2d: List[Tuple[int,int]] = [(6,6), (8,8), (12,12)], use_calibrated_bias: bool = True, seed: int = DEFAULT_SEED, time_budget_sec: Optional[float] = 300.0, out_csv: Optional[str] = None) -> pd.DataFrame:
    s_age, s_pr = load_two_numeric(dataset_path, age_col, priors_col)
    dfs: List[pd.DataFrame] = []
    print(f"[2D suite] rows={len(s_age)} cols=({age_col},{priors_col}) from {dataset_path}")

    dfs.append(run_2d_full(s_age, s_pr, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))
    for (na, nb) in discretized_bins_2d:
        dfs.append(run_2d_grouped_start(s_age, s_pr, nbins_age=na, nbins_pr=nb, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))
        dfs.append(run_2d_collapsed(s_age, s_pr, nbins_age=na, nbins_pr=nb, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))
        dfs.append(run_2d_cells_optimize_pair_eval(s_age, s_pr, nbins_age=na, nbins_pr=nb, use_calibrated_bias=use_calibrated_bias, seed=seed, time_budget_sec=time_budget_sec))

    out = pd.concat(dfs, ignore_index=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"[suite_2d] wrote {out_csv}")
    return out


# =============================================================================
# Plotting
# =============================================================================
def _series_styles():
    markers = ['o','s','^','D','v','P','X','h','>','<']
    dashes  = [(None,None), (5,2), (3,2), (2,2), (7,3), (9,2), (2,1), (4,1,1,1)]
    return markers, dashes

def plot_scalability_file(csv_path: str, *, metric: str = "time", utility_metric: str = "util_theta", x_mode: str = "auto", combine_qstar_time: bool = True, ylog: bool = False, save_png: Optional[str] = None, title: Optional[str] = None) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[plot] empty CSV.")
        return

    if x_mode == "bins":
        x = np.where(df["mode"].str.contains("2d", na=False),
                     df["nbins_age"].astype(float) * df["nbins_priors"].astype(float),
                     df["nbins"].astype(float))
        x_label = "#bins (1D) or #cells (2D)"
    elif x_mode == "domain":
        x = df["domain_size"].astype(float); x_label = "domain size"
    elif x_mode == "grid":
        x = df["grid_cells"].astype(float); x_label = "#present cells"
    else:
        x = np.where(df["grid_cells"].notna(), df["grid_cells"].astype(float),
             np.where(df["nbins"].notna(), df["nbins"].astype(float),
             df["domain_size"].astype(float)))
        x_label = "bins/cells (or domain size)"

    df = df.assign(x=x)

    if metric == "time":
        df["y"] = np.where(
            df["label"]=="baseline",
            df["time_baseline"],
            np.where(
                df["label"]=="q_base", df["time_qbase"],
                (df["time_qbase"] + df["time_mi"]) if combine_qstar_time else df["time_mi"]
            )
        )
        y_label = "Time (s)"; default_title = "Time vs Complexity"
    elif metric == "utility":
        df["y"] = df[utility_metric].astype(float)
        y_label = utility_metric; default_title = f"{utility_metric} vs Complexity"
    else:
        raise ValueError("metric must be 'time' or 'utility'")

    df = df.sort_values(["x","mode","bias_kind","label"])

    series_groups = list(df.groupby(["mode","bias_kind","label"], dropna=False))
    markers, dashes = _series_styles()
    plt.figure(figsize=(10,6))
    unique_series = len(series_groups)
    offsets = np.linspace(-0.15, 0.15, unique_series) if unique_series > 1 else [0.0]

    for idx, ((mode,bk,lbl), g) in enumerate(series_groups):
        x_vals = g["x"].values.astype(float); y_vals = g["y"].values.astype(float)
        x_dodged = x_vals + offsets[idx]
        style = markers[idx % len(markers)]; dash = dashes[idx % len(dashes)]
        (line,) = plt.plot(x_dodged, y_vals, marker=style, linewidth=2, label=f"{mode} · {bk} · {lbl}")
        if dash != (None,None): line.set_dashes(dash)

    if ylog: plt.yscale("log")
    plt.xlabel(x_label); plt.ylabel(y_label); plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=9, ncol=2)
    plt.title(title or default_title); plt.tight_layout()
    if save_png:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=150, bbox_inches="tight"); print(f"[plot] saved: {save_png}"); plt.close()
    else:
        plt.show()


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    DATASET_PATH = "../data/real/COMPAS.csv"
    AGE_COL      = "age"
    PRIORS_COL   = "priors_count"

    TIME_BUDGET_SEC = 300.0

    df_1d = run_scalability_suite_1d(
        DATASET_PATH, AGE_COL,
        discretized_bins=[8, 16, 32],    # fewer, cleaner
        use_calibrated_bias=True,
        time_budget_sec=TIME_BUDGET_SEC,
        out_csv="../results/scalability_1d.csv"
    )
    print("\n[1D RESULTS]\n", df_1d)

    df_2d = run_scalability_suite_2d(
        DATASET_PATH, AGE_COL, PRIORS_COL,
        discretized_bins_2d=[(6,6), (8,8)],  # keep modest
        use_calibrated_bias=True,
        time_budget_sec=TIME_BUDGET_SEC,
        out_csv="../results/scalability_2d.csv"
    )
    print("\n[2D RESULTS]\n", df_2d)

    plot_scalability_file("../results/scalability_1d.csv",  metric="utility", utility_metric="util_theta",
                          save_png="../results/plot_1d_util_theta.png")
    plot_scalability_file("../results/scalability_1d.csv",  metric="time",
                          save_png="../results/plot_1d_time.png")
    plot_scalability_file("../results/scalability_2d.csv",  metric="utility", utility_metric="util_theta",
                          save_png="../results/plot_2d_util_theta.png")
    plot_scalability_file("../results/scalability_2d.csv",  metric="time",
                          save_png="../results/plot_2d_time.png")
