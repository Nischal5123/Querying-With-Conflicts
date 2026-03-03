# -*- coding: utf-8 -*-
"""
COI (distinct-domain) + Scalability experiment over #attributes.

What you get:
- Core algorithms:
    * Alg.1  : Credibility detection (fast path for uniform+threshold)
    * Alg.2  : Build q_base by merging unsupported boundaries
    * Alg.4  : Maximally-informative DP over q_base
- Utilities:
    * evaluate_plan_utility(...) for q_babble/q_base/q_star
- Tuple-safe bias:
    * make_random_multilevel_bias(...) works for tuple domains (Cartesian product)
- Experiment:
    * run_scalability_time_vs_num_attrs_logging(...)
      - For k = 1..K: pick k columns (smallest domains first)
      - Build Cartesian domain of distinct values across those columns
      - Bias levels picked deterministically from the posterior grid {j/(|domain|+1)}
        (no grid search), probs fixed; seed fixed.
      - Measure:
          time_alg1_first_s, |C| from first pass,
          time_alg2_s, time_alg4_s, time_alg24_s (=2+4),
          utilities for q_babble/q_base/q_star, kept counts,
          q_base_groups, q_star_groups, merges_alg2
      - Append a row to CSV after each k
      - Plot (#attrs vs combined (2+4) time)

Default receiver: THRESHOLD, prior: UNIFORM by rank.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
import itertools, json, os, time, math
import csv as _csv


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Global tolerances
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-5    # numeric comparison tolerance
EPS_ORDER   = 1e-9    # tiny order-aware bonus weight (presentational/tie-break)

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
    Works with non-numeric domain values (e.g., tuples).
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
# Bias models (per-value bias)
# =============================================================================

@dataclass
class Bias1D:
    """
    Per-value biases. 'custom' lets you work on tuple domains.
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
        if self.kind == "custom" and self.custom:
            b = float(self.custom(x, domain_info))
            return float(min(max(b, 0.0), 1.0))

        # Numeric kinds try to normalize; if that fails, fall back to base.
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
                if t <= xs[0]:
                    b = ys[0]
                elif t >= xs[-1]:
                    b = ys[-1]
                else:
                    i = np.searchsorted(xs, t) - 1
                    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
                    w = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
                    b = (1 - w) * y0 + w * y1
        else:
            b = self.base
        return float(min(max(b, 0.0), 1.0))

@dataclass
class CompositeBias:
    rules: List[Bias1D]
    combine: str = "max"  # 'max' or 'sum'

    def bias_for_value(self, x: Any, domain_info: Dict[str, float]) -> float:
        vals = [r.bias_for_value(x, domain_info) for r in self.rules]
        if not vals:
            return 0.0
        if self.combine == "sum":
            return float(min(max(sum(vals), 0.0), 1.0))
        return float(min(max(max(vals), 0.0), 1.0))

def biases_from_bias_obj(domain: List[Any], bias_obj: Any) -> Dict[Any, float]:
    if not domain:
        return {}
    info = {"min": min(domain), "max": max(domain)}  # OK for tuples (lex order)
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}

def make_random_multilevel_bias(
    domain_values: List[Any],
    levels: Tuple[float, ...] = (0.6, 0.5, 0.2, 0.2),
    probs:  Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    seed: Optional[int] = 123,
) -> Bias1D:
    """Tuple-safe random multilevel bias: assigns each distinct value a level."""
    rng = np.random.default_rng(seed)
    dom = list(dict.fromkeys(domain_values))  # stable dedupe
    P = np.array(probs, dtype=float); P = P / P.sum()
    labels = rng.choice(len(levels), size=len(dom), p=P)
    value2bias = {dom[i]: float(levels[int(lbl)]) for i, lbl in enumerate(labels)}
    def f(value: Any, _info: Dict[str, float]) -> float:
        return value2bias.get(value, 0.0)
    return Bias1D(kind="custom", custom=f)

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

def system_best_response_quadratic(
    q_groups: List[List[Any]], domain: List[Any],
    post: Dict[Any, float], biases: Dict[Any, float],
    gamma: float = 1.0,
) -> Dict[Any, float]:
    """Quadratic receiver: action(v) = clip(gamma * post[v] + bias[v], 0, 1)."""
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
    if n == 0:
        return {}
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
        act = response  # 0/1
    else:
        base = -sum((response.get(v, 0.0) - theta.get(v, 0.0)) ** 2 for v in theta.keys())
        act = response  # [0,1]
    if eps_order > 0.0 and order_list is not None:
        w = _exposure_weights(order_list, scheme=exposure_scheme)
        bonus = sum(w.get(v, 0.0) * theta.get(v, 0.0) * float(act.get(v, 0.0)) for v in order_list)
        return base + eps_order * bonus
    return base

# =============================================================================
# Small helpers
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
        raise ValueError("q_groups must be a partition of `domain` (cover each value exactly once, no dups).")

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
    tie_policy: str = "auto",          # "auto" | "neutral" | "force_on_ties"
    rule: str = "paper",               # "paper" | "strict_opposite"
) -> Set[Tuple[Any, Any]]:
    """
    Compute set C of credible strict edges present in q_groups.

    Fast path (uniform prior + threshold + eps≈0):
      swapping (u,v) only reassigns group-average posteriors; O(1) deltas.

    rule:
      - "paper": mark (u,v) unless BOTH types strictly prefer the same column
      - "strict_opposite": require strict & opposite choices across types
    """
    if tie_policy == "auto" and _is_fully_symmetric_bias(biases):
        return set(op_pairs_strict(q_groups))

    fast = (prior.kind == "uniform" and receiver_model == "threshold" and abs(eps_order) <= 1e-12)
    if fast:
        items = [x for g in q_groups for x in g]
        k = len(domain)
        idx = {v: i for i, v in enumerate(items)}

        gid_of = [None] * len(items)
        for gi, g in enumerate(q_groups):
            for v in g:
                gid_of[idx[v]] = gi

        a_g = []
        r = 1
        for g in q_groups:
            n = len(g)
            j_hi = k - r + 1
            j_lo = k - (r + n - 1) + 1
            a_g.append(((j_hi + j_lo) / 2.0) / (k + 1.0))
            r += n

        theta_plus = [a_g[gid_of[i]] for i in range(len(items))]
        beta_q = [1 if (theta_plus[i] - biases[items[i]]) > EPS_COMPARE else 0
                  for i in range(len(items))]
        U_pp = sum(theta_plus[i] * beta_q[i] for i in range(len(items)))

        C: Set[Tuple[Any, Any]] = set()
        for gi in range(len(q_groups)):
            for gj in range(gi + 1, len(q_groups)):
                ai, aj = a_g[gi], a_g[gj]
                for u in q_groups[gi]:
                    iu = idx[u]; bu = biases[u]
                    betas_u = 1 if (aj - bu) > EPS_COMPARE else 0
                    for v in q_groups[gj]:
                        iv = idx[v]; bv = biases[v]
                        betas_v = 1 if (ai - bv) > EPS_COMPARE else 0

                        up_q = U_pp
                        up_s = U_pp \
                             + theta_plus[iu] * (betas_u - beta_q[iu]) \
                             + theta_plus[iv] * (betas_v - beta_q[iv])

                        um_q = U_pp \
                             + (aj - theta_plus[iu]) * beta_q[iu] \
                             + (ai - theta_plus[iv]) * beta_q[iv]

                        um_s = up_s \
                             + (aj - theta_plus[iu]) * betas_u \
                             + (ai - theta_plus[iv]) * betas_v

                        if rule == "paper":
                            both_q    = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
                            both_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
                            if not (both_q or both_swap):
                                C.add((u, v))
                        else:
                            cplus_q  = (up_q > up_s + EPS_COMPARE)
                            cplus_sw = (up_s > up_q + EPS_COMPARE)
                            cminus_q  = (um_q > um_s + EPS_COMPARE)
                            cminus_sw = (um_s > um_q + EPS_COMPARE)
                            if (cplus_q and cminus_sw) or (cplus_sw and cminus_q):
                                C.add((u, v))
        return C

    # Generic fallback
    C: Set[Tuple[Any, Any]] = set()

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

        if rule == "paper":
            both_q    = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
            both_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
            if not (both_q or both_swap):
                C.add((u, v))
        else:
            cplus  = _decide(up_q, up_s)
            cminus = _decide(um_q, um_s)
            if tie_policy == "force_on_ties":
                if cplus is None:  cplus  = "q"
                if cminus is None: cminus = "swap"
            if cplus is not None and cminus is not None and cplus != cminus:
                C.add((u, v))
    return C

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

def _boundary_supported(idx_map, R, Gi: List[Any], Gj: List[Any],
                        *, policy: str = "all", alpha: float = 0.5) -> bool:
    """
    policy:
      - "all":      every (u,v) with u∈Gi, v∈Gj is reachable (original)
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

def algorithm_2_build_qbase(
    initial_order: List[Any], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0,
    eps_order: float = 0.0, rule: str = "paper",
    boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[Any]]:
    q_cur: List[List[Any]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(
            q_cur, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma, eps_order=0.0, rule=rule
        )
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported(idx_map, R, Gi, Gj, policy=boundary_policy, alpha=alpha):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur

def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
    biases: Dict[Any, float], receiver_model: str = "threshold", gamma: float = 1.0,
    eps_order: float = 0.0, rule: str = "paper",
    boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[Any]]:
    _validate_partition_or_die(q_init_groups, domain)
    q_cur = [g[:] for g in q_init_groups]
    for _ in range(max(0, len(q_cur) - 1) + 1):
        C = algorithm_1_credibility_detection(
            q_cur, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma, eps_order=0.0, rule=rule
        )
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported(idx_map, R, Gi, Gj, policy=boundary_policy, alpha=alpha):
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
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order_for_tiebreak: float = 0.0
) -> List[List[Any]]:
    m = len(q_base)
    if m == 0:
        return []

    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model, gamma=gamma)
    base_gain0: List[float] = []
    base_gain_eps: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        resp_t = {x: beta_base.get(x, 0 if receiver_model == "threshold" else 0.0) for x in items_t}
        u0   = user_utility_from_response(theta, resp_t, receiver_model, order_list=None,    eps_order=0.0)
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

            u0_run   = user_utility_from_response(theta, resp_run, receiver_model, order_list=None,     eps_order=0.0)
            ueps_run = user_utility_from_response(theta, resp_run, receiver_model, order_list=run_items, eps_order=eps_order_for_tiebreak)

            base_line0  = sum(base_gain0[t]   for t in range(i, j+1))
            base_lineep = sum(base_gain_eps[t] for t in range(i, j+1))
            C0[i][j]   = u0_run   - base_line0
            Ceps[i][j] = ueps_run - base_lineep

    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    dp   = [0.0] * (m + 1)
    tie  = [0.0] * (m + 1)
    prev = [-1]  * (m + 1)

    for t in range(1, m + 1):
        best_val, best_tie, arg = -float("inf"), -float("inf"), -1
        for i in range(1, t + 1):
            g0   = C0[i-1][t-1]
            geps = Ceps[i-1][t-1]
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

# =============================================================================
# Utilities for reporting and evaluation
# =============================================================================

def evaluate_plan_utility(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
) -> Tuple[float, Optional[int]]:
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta = system_best_response(q_groups, domain, prior, biases,
                                receiver_model=receiver_model, gamma=gamma)
    util = user_utility_from_response(theta, beta, receiver_model, order_list=None, eps_order=0.0)
    kept = None
    if receiver_model == "threshold":
        kept = sum(int(beta.get(v, 0)) for v in (x for g in q_groups for x in g))
    return float(util), kept

# =============================================================================
# Experiment helpers (Cartesian domain over attributes)
# =============================================================================

def _distinct_values_sorted(df: pd.DataFrame, col: str):
    xs = pd.unique(df[col])
    try:
        return sorted(xs.tolist())
    except Exception:
        return sorted(map(str, xs.tolist()))

def build_cartesian_domain(df: pd.DataFrame, cols: List[str]) -> List[Any]:
    """
    Domain = Cartesian product of DISTINCT values across given columns.
    Items are tuples; final order is DESC lexicographic (to define ranks).
    """
    domains = [_distinct_values_sorted(df, c) for c in cols]
    prod = itertools.product(*domains)
    dom = [tuple(t) for t in prod]
    dom.sort(reverse=True)
    return dom

def pick_columns_smallest_domains(df: pd.DataFrame, candidates: List[str], k: int) -> List[str]:
    stats = sorted([(c, int(df[c].nunique())) for c in candidates], key=lambda t: (t[1], t[0]))
    return [c for c, _ in stats[:k]]

# =============================================================================
# Bias levels from posteriors (no search)
# =============================================================================

def posterior_levels_for_size(n: int) -> Tuple[float, float, float, float]:
    """
    Choose 4 nonincreasing bias levels from the posterior grid {j/(n+1)}.
    We take ranks at ~100%, 75%, 50%, 25% (rounded up to valid ranks).
    """
    if n <= 0:
        return (0.5, 0.5, 0.5, 0.5)
    ranks = [
        n,
        max(1, math.ceil(0.75 * n)),
        max(1, math.ceil(0.50 * n)),
        max(1, math.ceil(0.25 * n)),
    ]
    levels = [r / (n + 1.0) for r in ranks]
    levels.sort(reverse=True)
    return tuple(levels)

# =============================================================================
# Scalability experiment: time vs #attributes (stores rich metrics)
# =============================================================================

def run_scalability_time_vs_num_attrs_logging(
    df: pd.DataFrame,
    candidate_cols: List[str],
    *,
    max_k: Optional[int] = None,
    total_timeout_s: float = 600.0,       # 10 minutes
    csv_path: str = "../results/tables/time_vs_num_attrs.csv",
    plot_path: str = "../results/plots/time_vs_num_attrs.png",
    append: bool = False,                 # overwrite by default
    rule: str = "paper",
    boundary_policy: str = "quantile",    # match "working" setup
    alpha: float = 0.5,                   # match "working" setup
    bias_levels: Optional[Tuple[float, ...]] = None,   # None => posterior-based
    bias_probs:  Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    bias_seed: int = 123,
    print_progress: bool = True,
):
    """
    For k = 1..max_k:
      - choose k columns (smallest domain sizes first),
      - build full Cartesian domain over those columns (tuples),
      - bias = random multilevel over the Cartesian domain,
        levels picked deterministically from {j/(|domain|+1)} unless provided,
      - measure & log:
          time_alg1_first_s, num_edges_alg1 (=|C| for first-pass singletons),
          time_alg2_s, time_alg4_s, time_alg24_s (=2+4),
          q_base_groups, q_star_groups, merges_alg2,
          utilities & kept for q_babble/q_base/q_star.
      - write a row to CSV after each k
      - plot (#attrs vs time_alg24_s)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    header = [
        "timestamp", "k_attrs", "cols_json", "domain_size",
        "levels_json", "probs_json",
        "time_alg1_first_s", "num_edges_alg1",
        "time_alg2_s", "time_alg4_s", "time_alg24_s",
        "q_base_groups", "q_star_groups", "merges_alg2",
        "utility_q_babble", "utility_q_base", "utility_q_star",
        "kept_q_babble", "kept_q_base", "kept_q_star",
    ]

    want_header = True
    if append and os.path.exists(csv_path):
        want_header = False
    if (not append) and os.path.exists(csv_path):
        os.remove(csv_path)

    f = open(csv_path, "a", newline="")
    w = _csv.writer(f, quoting=_csv.QUOTE_NONNUMERIC, escapechar="\\")
    if want_header:
        w.writerow(header); f.flush()

    start_all = time.perf_counter()
    prior = PriorSpec(kind="uniform")

    ordered_cols = pick_columns_smallest_domains(df, candidate_cols, len(candidate_cols))
    if max_k is None:
        max_k = len(ordered_cols)

    xs, ys_alg24, ys_alg1 = [], [], []
    for k in range(1, max_k + 1):
        cols_k = ordered_cols[:k]
        dom = build_cartesian_domain(df, cols_k)
        print(f"[k={k}] cols={cols_k} => |domain|={len(dom):,}")
        ksize = len(dom)
        if ksize == 0:
            if print_progress:
                print(f"[k={k}] EMPTY domain for cols={cols_k}; skipping.")
            continue

        levels_to_use = bias_levels if bias_levels is not None else posterior_levels_for_size(ksize)
        bias_obj = make_random_multilevel_bias(dom, levels=levels_to_use, probs=bias_probs, seed=bias_seed)
        biases   = biases_from_bias_obj(dom, bias_obj)

        # Alg.1 (first pass on singletons)
        q_singletons = [[v] for v in dom]
        t1a = time.perf_counter()
        C0 = algorithm_1_credibility_detection(
            q_singletons, dom, prior,
            biases=biases, receiver_model="threshold",
            gamma=1.0, eps_order=0.0,
            tie_policy="neutral", rule=rule
        )
        t1b = time.perf_counter()
        time_alg1_first = t1b - t1a
        num_edges_alg1 = len(C0)

        # Alg.2
        t2a = time.perf_counter()
        q_base = algorithm_2_build_qbase(
            dom, dom, prior,
            biases=biases, receiver_model="threshold", gamma=1.0,
            eps_order=EPS_ORDER, rule=rule,
            boundary_policy=boundary_policy, alpha=alpha
        )
        t2b = time.perf_counter()
        time_alg2 = t2b - t2a

        q_base_groups = len(q_base)
        merges_alg2   = len(dom) - q_base_groups

        # Alg.4
        t4a = time.perf_counter()
        q_star = algorithm_4_maximally_informative(
            q_base, dom, prior,
            biases=biases, receiver_model="threshold", gamma=1.0,
            eps_order_for_tiebreak=0.0
        )
        t4b = time.perf_counter()
        time_alg4 = t4b - t4a

        time_alg24 = time_alg2 + time_alg4

        util_babble, kept_babble = evaluate_plan_utility([dom[:]], dom, prior, biases=biases, receiver_model="threshold", gamma=1.0)
        util_base,   kept_base   = evaluate_plan_utility(q_base,    dom, prior, biases=biases, receiver_model="threshold", gamma=1.0)
        util_star,   kept_star   = evaluate_plan_utility(q_star,    dom, prior, biases=biases, receiver_model="threshold", gamma=1.0)

        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            k,
            json.dumps(cols_k),
            ksize,
            json.dumps(tuple(map(float, levels_to_use))),
            json.dumps(tuple(map(float, bias_probs))),
            time_alg1_first,
            num_edges_alg1,
            time_alg2,
            time_alg4,
            time_alg24,
            q_base_groups,
            len(q_star),
            merges_alg2,
            util_babble,
            util_base,
            util_star,
            kept_babble,
            kept_base,
            kept_star,
        ]
        w.writerow(row); f.flush()

        if print_progress:
            print(f"[k={k}] cols={cols_k} | |domain|={ksize:,} | "
                  f"Alg1_first={time_alg1_first:.3f}s (|C|={num_edges_alg1:,}) | "
                  f"Alg2={time_alg2:.3f}s | Alg4={time_alg4:.3f}s | 2+4={time_alg24:.3f}s | "
                  f"q_base={q_base_groups}→q*={len(q_star)} | "
                  f"U(babble)={util_babble:.4g}, U(base)={util_base:.4g}, U(q★)={util_star:.4g}")

        xs.append(k); ys_alg24.append(time_alg24); ys_alg1.append(time_alg1_first)

        if (time.perf_counter() - start_all) > total_timeout_s:
            if print_progress:
                print(f"⏱️  Stopping at k={k} due to total timeout ({total_timeout_s}s).")
            break

    f.close()

    # Plot
    if xs:
        plt.figure(figsize=(5.2, 3.0))
        ys_alg1_ms  = [y * 1000 for y in ys_alg1]
        ys_alg24_ms = [y * 1000 for y in ys_alg24]
        plt.plot(xs, ys_alg1_ms,  marker="x", linestyle="-", label="Credibility Detection")
        plt.plot(xs, ys_alg24_ms, marker="o", linestyle="-", label="Maximally Informative (2+4)")
        plt.xlabel("Number of attributes")
        plt.ylabel("Execution time (ms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()


# =============================================================================
# Example CLI
# =============================================================================
def load_flights_df():
    candidate = "../data/real/flights_bucketized.csv"
    if os.path.exists(candidate):
        try:
            df = pd.read_csv(candidate)
            return df, True
        except Exception:
            pass
    # Synthetic fallback
    rng = np.random.default_rng(123)
    n = 5000
    df = pd.DataFrame({
        "age": rng.integers(18, 80, size=n),
        "education_num": rng.integers(1, 17, size=n),
        "sex": rng.choice(["Male", "Female"], size=n, p=[0.51, 0.49])
    })
    return df, False

def encode_attributes(df: pd.DataFrame):
    df = df.copy()
    # Ensure numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["education_num"] = pd.to_numeric(df["education_num"], errors="coerce")
    # Encode sex to 0/1 (Female=0, Male=1)
    sex_map = {"Female": 0, "F": 0, 0: 0, "Male": 1, "M": 1, 1: 1}
    df["sex"] = df["sex"].map(sex_map).fillna(0).astype(int)
    df = df.dropna(subset=["age", "education_num", "sex"])
    return df

if __name__ == "__main__":
    # Minimal demo with a synthetic or bucketized dataset.
    dataset = "flights"
    df, found = load_flights_df()

    candidate_cols = ["airline", "price_category", "days_category"]
    useless = ["education"]  # keep if present in some real CSVs; drop if irrelevant
    candidate_cols = [c for c in candidate_cols if c not in useless]

    run_scalability_time_vs_num_attrs_logging(
        df,
        candidate_cols=candidate_cols,
        max_k=len(candidate_cols),
        total_timeout_s=6000.0,  # 3 hours budget
        csv_path=f'../results/tables/{dataset}_3_time_vs_num_attrs.csv',
        plot_path=f'../results/plots/{dataset}_3_time_vs_num_attrs.png',
        append=False,
        rule="paper",
        boundary_policy="quantile",   # match working setup
        alpha=0.5,
        bias_levels=None,             # derive from posterior grid (no search)
        bias_probs=(0.25, 0.25, 0.25, 0.25),
        bias_seed=123,
        print_progress=True,
    )

    print(f"Done. CSV -> ../results/tables/{dataset}_3_time_vs_num_attrs.csv")
    print(f"Plot -> ../results/plots/{dataset}_3_time_vs_num_attrs.png")
