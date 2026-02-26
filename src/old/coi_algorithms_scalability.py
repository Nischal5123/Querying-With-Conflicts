# -*- coding: utf-8 -*-
"""
Core COI algorithms with small, opt-in instrumentation for scalability experiments.

What this file provides:
- PriorSpec and posterior computation for ordered groups (ties allowed).
- Bias models (with a robust 'custom' path that supports tuple domains).
- Receiver models: threshold / quadratic.
- Utility with optional tiny presentation bonus (used only for tie-breaks).
- Algorithms:
    • Alg. 1: credibility detection over OP(q) (strict inter-group pairs).
    • Alg. 2: build q_base by merging unsupported adjacent group boundaries.
      - Two variants:
          (a) start from singletons (strict total order)
          (b) start from a user partial ranking (ties honored; merge-only)
      - NEW: optional `timing` dict to accumulate Alg.1 time and iterations.
    • Alg. 4: maximally informative dynamic program (merge contiguous runs).
- Pipeline helpers (unchanged behavior) and reporting.

Safety & robustness:
- 'custom' bias now short-circuits and never attempts numeric casts,
  so tuple domains (e.g., (age, priors)) are fully supported.
- Numeric kinds only normalize when domain is numeric; otherwise they
  harmlessly fall back to `base`.

NOTE: We do NOT change return signatures; Alg. 2 simply accepts an optional
`timing: dict` to write timing data. If omitted, behavior is identical to before.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
from collections import deque
from time import perf_counter
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Global tolerances (keep tiny order bonus for tie-breaking only)
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-5
EPS_ORDER   = 1e-9

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
    custom: Optional[Callable[[int, int], float]] = None

    def rank_expectation(self, k: int, r_desc: int) -> float:
        if k <= 0 or r_desc <= 0 or r_desc > k:
            return 0.0
        if self.kind == "uniform":
            j = k - r_desc + 1
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
        # default back to uniform
        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec
) -> Dict[float, float]:
    """
    For an ordered partition q_groups of the DISTINCT domain:
      - Each group occupies a consecutive block of ranks.
      - All values in a group get the average expectation across those ranks.
    """
    k = len(domain)
    post: Dict[float, float] = {}
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
        for v in g: post[v] = float(a)
        r += n
    for v in domain: post.setdefault(v, 0.0)
    return post

# =============================================================================
# Bias models
# =============================================================================

@dataclass
class Bias1D:
    """
    For numeric domains, several built-in shapes are provided. For non-numeric
    domains (e.g., tuples), ALWAYS use kind='custom' — which now short-circuits
    before any numeric casts.
    """
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
    custom: Optional[Callable[[Any, Dict[str, float]], float]] = None

    def _norm(self, x: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    def bias_for_value(self, x, domain_info: Dict[str, float]) -> float:
        """
        'custom': early return (works with tuples; no numeric casts).
        Numeric kinds: attempt normalization; if domain isn't numeric,
        fall back to base (clamped) instead of crashing.
        """
        if self.kind == "custom" and self.custom:
            b = float(self.custom(x, domain_info))
            return float(min(max(b, 0.0), 1.0))

        # Try to compute normalized coordinate t for numeric kinds
        try:
            lo = float(domain_info.get("min", 0.0))
            hi = float(domain_info.get("max", 1.0))
            xv = float(x)
            t = self._norm(xv, lo, hi)
        except (TypeError, ValueError):
            return float(min(max(self.base, 0.0), 1.0))

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


def biases_from_bias_obj(domain: List[Any], bias_obj: Any) -> Dict[Any, float]:
    """Materialize bias per domain element. Supports tuples via 'custom'."""
    if not domain: return {}
    info = {"min": min(domain), "max": max(domain)} if all(isinstance(v, (int,float)) for v in domain) else {}
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}

# =============================================================================
# Receiver & utility
# =============================================================================

def system_best_response_threshold(
    q_groups: List[List[Any]], domain: List[Any],
    post: Dict[Any, float], biases: Dict[Any, float],
) -> Dict[Any, int]:
    return {v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0
            for g in q_groups for v in g}

def system_best_response_quadratic(
    q_groups: List[List[Any]], domain: List[Any],
    post: Dict[Any, float], biases: Dict[Any, float], gamma: float = 1.0,
) -> Dict[Any, float]:
    out: Dict[Any, float] = {}
    for g in q_groups:
        for v in g:
            a = gamma * post.get(v, 0.0) + biases.get(v, 0.0)
            out[v] = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return out

def system_best_response(
    q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0,
):
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    if receiver_model == "quadratic":
        return system_best_response_quadratic(q_groups, domain, post, biases, gamma=gamma)
    raise ValueError("receiver_model must be 'threshold' or 'quadratic'")

def _exposure_weights(order_list: List[Any], scheme: str = "harmonic") -> Dict[Any, float]:
    n = len(order_list)
    if n == 0: return {}
    if scheme == "harmonic":
        raw = [1.0 / (r + 1) for r in range(n)]
    elif scheme == "geometric":
        raw = [0.9 ** r for r in range(n)]
    else:
        raw = [1.0 / (r + 1) for r in range(n)]
    Z = sum(raw) or 1.0
    ws = [w / Z for w in raw]
    return {order_list[r]: ws[r] for r in range(n)}

def user_utility_from_response(
    theta: Dict[Any, float], response, receiver_model: str,
    *, order_list: Optional[List[Any]] = None, eps_order: float = 0.0, exposure_scheme: str = "harmonic",
) -> float:
    if receiver_model == "threshold":
        base = sum(theta.get(v, 0.0) * a for v, a in response.items())
        act = response
    else:
        base = -sum((response.get(v, 0.0) - theta.get(v, 0.0)) ** 2 for v in theta.keys())
        act = response
    if eps_order > 0.0 and order_list is not None:
        w = _exposure_weights(order_list, scheme=exposure_scheme)
        bonus = sum(w.get(v, 0.0) * theta.get(v, 0.0) * float(act.get(v, 0.0)) for v in order_list)
        return base + eps_order * bonus
    return base

# =============================================================================
# Helpers
# =============================================================================

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

def dedupe_and_sort_desc(values: List[Any]) -> List[Any]:
    return sorted(set(values), reverse=True)

def _validate_partition_or_die(q_groups: List[List[Any]], domain: List[Any]) -> None:
    flat = [x for g in q_groups for x in g]
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_groups must partition `domain` (cover each element exactly once).")

# =============================================================================
# Alg. 1, 2, 4
# =============================================================================

def _is_fully_symmetric_bias(biases: Dict[Any, float]) -> bool:
    if not biases:
        return True
    vals = list(biases.values())
    return (max(vals) - min(vals)) <= EPS_COMPARE

def algorithm_1_credibility_detection(
    q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order: float = 0.0,
    tie_policy: str = "auto",
    cred_rule: str = "not_both_same",
) -> Set[Tuple[Any, Any]]:
    """
    Compute credible edges among OP(q) (inter-group strict pairs).
    We use the “strict & opposite” rule (your original).
    """
    C: Set[Tuple[Any, Any]] = set()

    if tie_policy == "auto" and _is_fully_symmetric_bias(biases):
        for (u, v) in op_pairs_strict(q_groups):
            C.add((u, v))
        return C

    def _decide(a: float, b: float) -> Optional[str]:
        if a > b + EPS_COMPARE: return "q"
        if b > a + EPS_COMPARE: return "swap"
        return None

    theta_plus = compute_expected_posteriors(q_groups, domain, prior)
    beta_q     = system_best_response(q_groups, domain, prior, biases, receiver_model, gamma=gamma)
    order_q    = flatten_in_order(q_groups)

    for (u, v) in op_pairs_strict(q_groups):
        q_swap    = swap_single_pair(q_groups, u, v)
        theta_min = compute_expected_posteriors(q_swap, domain, prior)
        beta_s    = system_best_response(q_swap, domain, prior, biases, receiver_model, gamma=gamma)
        order_s   = flatten_in_order(q_swap)

        up_q = user_utility_from_response(theta_plus, beta_q, receiver_model, order_list=order_q, eps_order=eps_order)
        up_s = user_utility_from_response(theta_plus, beta_s, receiver_model, order_list=order_s, eps_order=eps_order)
        um_q = user_utility_from_response(theta_min,  beta_q, receiver_model, order_list=order_q, eps_order=eps_order)
        um_s = user_utility_from_response(theta_min,  beta_s, receiver_model, order_list=order_s, eps_order=eps_order)

        # strict-compare helper:

        cplus = _decide(up_q, up_s)
        cminus = _decide(um_q, um_s)

        if cred_rule == "strict_opposite":
            if cplus is not None and cminus is not None and cplus != cminus:
                C.add((u, v))
        elif cred_rule == "not_both_same":
            both_prefer_q = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
            both_prefer_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
            if not (both_prefer_q or both_prefer_swap):
                C.add((u, v))
        else:
            raise ValueError("cred_rule must be 'strict_opposite' or 'not_both_same'")

    return C

def build_adj(items: List[Any], edges: Set[Tuple[Any, Any]]) -> Dict[Any, List[Any]]:
    adj = {x: [] for x in items}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)
    return adj

def reachable(start: Any, adj: Dict[Any, List[Any]]) -> Set[Any]:
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for w in adj.get(u, []):
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return seen

def algorithm_2_build_qbase(
    initial_order: List[Any], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0, eps_order: float = 0.0,
    timing: Optional[dict] = None
) -> List[List[Any]]:
    """
    Start from SINGLETONS and iteratively MERGE adjacent groups whose boundary
    is not supported by C-reachability.
    - If `timing` dict is given, we accumulate:
        timing['time_cred']  := total time spent inside Alg.1 across iterations
        timing['iterations'] := number of merge attempts
    """
    if timing is not None:
        timing.setdefault('time_cred', 0.0)
        timing.setdefault('iterations', 0)

    q_cur: List[List[Any]] = [[x] for x in initial_order]

    while True:
        print("Current q_cur size:", len(q_cur))
        if timing is not None: timing['iterations'] += 1
        t_cred0 = perf_counter()
        C = algorithm_1_credibility_detection(q_cur, domain, prior,
                                              biases=biases, receiver_model=receiver_model, gamma=gamma,
                                              eps_order=eps_order)
        print("Credible edges found:", len(C))

        if timing is not None:
            timing['time_cred'] += (perf_counter() - t_cred0)

        adj = build_adj(domain, C)
        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur

def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0, eps_order: float = 0.0,
    timing: Optional[dict] = None
) -> List[List[Any]]:
    """
    Start from USER PARTIAL RANKING (ties intact) and MERGE unsupported boundaries.
    Optional `timing` behaves like the singletons variant.
    """
    _validate_partition_or_die(q_init_groups, domain)

    if timing is not None:
        timing.setdefault('time_cred', 0.0)
        timing.setdefault('iterations', 0)

    q_cur: List[List[Any]] = [g[:] for g in q_init_groups]
    while True:
        if timing is not None: timing['iterations'] += 1
        t_cred0 = perf_counter()
        C = algorithm_1_credibility_detection(q_cur, domain, prior,
                                              biases=biases, receiver_model=receiver_model, gamma=gamma,
                                              eps_order=eps_order)
        if timing is not None:
            timing['time_cred'] += (perf_counter() - t_cred0)

        adj = build_adj(domain, C)
        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur

def algorithm_4_maximally_informative(
    q_base: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0,
    eps_order_for_tiebreak: float = 0.0
) -> List[List[Any]]:
    """
    DP over contiguous runs of q_base adding only positive base gains; eps-order score
    is used solely for tie-breaks (style points).
    """
    m = len(q_base)
    if m == 0: return []

    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model, gamma=gamma)
    base_gain0: List[float] = []
    base_gain_eps: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        resp_t = {x: beta_base.get(x, 0 if receiver_model == "threshold" else 0.0) for x in items_t}
        u0 = user_utility_from_response(theta, resp_t, receiver_model, order_list=None, eps_order=0.0)
        ueps = user_utility_from_response(theta, resp_t, receiver_model, order_list=items_t, eps_order=eps_order_for_tiebreak)
        base_gain0.append(u0)
        base_gain_eps.append(ueps)

    C0 = [[0.0] * m for _ in range(m)]
    Ceps = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            run_items = [x for g in q_base[i:j+1] for x in g]
            temp_q = q_base[:i] + [run_items] + q_base[j+1:]
            beta_run = system_best_response(temp_q, domain, prior, biases, receiver_model, gamma=gamma)
            resp_run = {x: beta_run.get(x, 0 if receiver_model == "threshold" else 0.0) for x in run_items}
            u0_run = user_utility_from_response(theta, resp_run, receiver_model, order_list=None, eps_order=0.0)
            ueps_run = user_utility_from_response(theta, resp_run, receiver_model, order_list=run_items, eps_order=eps_order_for_tiebreak)
            base_line0  = sum(base_gain0[t]   for t in range(i, j+1))
            base_lineep = sum(base_gain_eps[t] for t in range(i, j+1))
            C0[i][j]   = u0_run   - base_line0
            Ceps[i][j] = ueps_run - base_lineep

    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    dp, tie, prev = [0.0]*(m+1), [0.0]*(m+1), [-1]*(m+1)
    for t in range(1, m+1):
        best_val, best_tie, arg = -float("inf"), -float("inf"), -1
        for i in range(1, t+1):
            g0, geps = C0[i-1][t-1], Ceps[i-1][t-1]
            val  = dp[i-1] + (g0   if g0   > EPS_COMPARE else 0.0)
            tiev = tie[i-1] + (geps if g0   > EPS_COMPARE else 0.0)
            better = (val > best_val + EPS_COMPARE) or \
                     (abs(val - best_val) <= EPS_COMPARE and (tiev > best_tie + EPS_COMPARE)) or \
                     (abs(val - best_val) <= EPS_COMPARE and abs(tiev - best_tie) <= EPS_COMPARE and (i - 1) > arg)
            if better:
                best_val, best_tie, arg = val, tiev, i - 1
        dp[t], tie[t], prev[t] = best_val, best_tie, arg

    q_star: List[List[Any]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star
