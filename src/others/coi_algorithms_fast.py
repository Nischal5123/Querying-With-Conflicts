# -*- coding: utf-8 -*-
"""
Domain-distinct COI analysis with threshold & quadratic receivers + tests.

What this file does (high level):
- Works at the DISTINCT DOMAIN level (ties collapsed across duplicate rows).
- Computes expected posteriors by rank; tied groups share the same posterior (the
  average of ranks covered by that group).
- Supports per-value bias functions (threshold/quadratic receivers).
- Computes user utilities (base + optional tiny presentation bonus).
- Provides:
    • Alg. 1: Credibility detection on the strict edges present in a (possibly tied) query.
    • Alg. 2: Build q_base by MERGING adjacent groups not supported by credibility.
      (Two variants):
         (a) algorithm_2_build_qbase           – starts from singletons (strict order)
         (b) algorithm_2_build_qbase_from_groups – starts from a partial ranking (ties honored)
    • Alg. 4: Maximally informative merge DP over q_base.
- Includes a wrapper run_pipeline(...) that accepts OPTIONAL q_user (partial ranking)
  but keeps the old run_pipeline_domain(...) API working.

Key design choices preserved (minimal changes):
- We do NOT split user-provided ties (partial ranking is the strategy).
- Alg. 1 checks only inter-group (strict) pairs in the input query (consistent
  with comparative cheap talk interpretation).
- Alg. 2-from-groups only MERGES across unsupported boundaries (never splits).

Author notes:
- Comments are intentionally verbose to make the code a self-contained reference.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
from collections import deque
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import time
import os
import itertools
import csv as _csv

# ---------------------------------------------------------------------
# Global tolerances
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-5    # numeric comparison tolerance
EPS_ORDER   = 1e-9    # tiny order-aware bonus weight (presentational/tie-break)

# =============================================================================
# Priors (how rank maps to expected preference)
# =============================================================================
def evaluate_plan_utility(
    q_groups: List[List[float]],
    domain: List[float],
    prior: PriorSpec,
    *,
    biases: Dict[float, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
) -> Tuple[float, Optional[int]]:
    """
    Compute the user's base utility (no presentation bonus) for a given plan q_groups.
    Returns (utility, kept_count_or_None).
    - For 'threshold', also returns the number of kept items.
    - For 'quadratic', kept_count is None.
    """
    # θ aligned to singleton domain (consistent with Alg.4 scoring)
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    # System's best response to the presented plan
    beta = system_best_response(q_groups, domain, prior, biases,
                                receiver_model=receiver_model, gamma=gamma)

    # Base utility only (no order epsilon)
    util = user_utility_from_response(theta, beta, receiver_model,
                                      order_list=None, eps_order=0.0)

    kept = None
    if receiver_model == "threshold":
        kept = sum(int(beta.get(v, 0)) for v in (x for g in q_groups for x in g))
    return float(util), kept

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
            # For uniform, the classical (rank/k+1) trick (j = k - r + 1).
            j = k - r_desc + 1
            return j / (k + 1.0)

        if self.kind == "beta":
            # Beta-shaped expectation across ranks.
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

        # Default: uniform fallback
        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec
) -> Dict[float, float]:
    """
    Compute expected posteriors for each domain value GIVEN the group ranks in q_groups.
    - q_groups is a list of groups (ties allowed).
    - All items in a group share the same posterior: the average of the expectations
      across the consecutive ranks the group occupies.
    - domain (DESC) defines k and (indirectly) the rank spectrum.
    """
    k = len(domain)
    post: Dict[float, float] = {}
    r = 1  # current group’s starting DESC rank

    for g in q_groups:
        n = len(g)
        if n == 0:
            continue

        if prior.kind == "uniform":
            # Fast closed-form for uniform: average of j/(k+1) over a block of size n
            # ranks r..r+n-1 map to j_hi..j_lo, average is ((j_hi + j_lo)/2)/(k+1).
            j_hi = k - r + 1
            j_lo = k - (r + n - 1) + 1
            a = ((j_hi + j_lo) / 2.0) / (k + 1.0)
        else:
            # General case: compute via numeric mean of rank_expectation.
            a = sum(prior.rank_expectation(k, rr) for rr in range(r, r + n)) / float(n)

        for v in g:
            post[v] = float(a)
        r += n

    # Safety: set any missing domain members to 0.0 (shouldn’t happen if q_groups is a partition)
    for v in domain:
        post.setdefault(v, 0.0)
    return post


# =============================================================================
# Bias models (per-value bias)
# =============================================================================

@dataclass
class Bias1D:
    """
    A convenient container to define per-value biases. Most kinds use the
    normalized location t = (x - min) / (max - min) unless using a "custom" fn.

    IMPORTANT:
    - For tuple / non-numeric domains (e.g., (age, priors)), ALWAYS use kind='custom'.
      This method now short-circuits for 'custom' and does not attempt numeric casts.
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
    custom: Optional[Callable[[float, Dict[str, float]], float]] = None

    def _norm(self, x: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    def bias_for_value(self, x, domain_info: Dict[str, float]) -> float:
        """
        Return a bias in [0,1] for the given domain element x.

        BEHAVIOR:
        - 'custom': early return (no numeric casts) so it works with tuple domains.
        - Numeric kinds ('linear_*', 'sigmoid', 'window', 'gaussian', 'piecewise', etc.)
          compute normalization ONLY if x, min, max are numeric; otherwise we fall back
          to the 'base' to avoid crashing a run that passed a non-numeric domain by mistake.
        """
        # 1) CUSTOM: do NOT touch numeric normalization; let the user function decide.
        if self.kind == "custom" and self.custom:
            b = float(self.custom(x, domain_info))
            return float(min(max(b, 0.0), 1.0))

        # 2) For numeric kinds we *try* to compute normalized location t.
        #    If the domain is non-numeric (e.g., tuples), fail gracefully to 'base'.
        try:
            lo = float(domain_info.get("min", 0.0))
            hi = float(domain_info.get("max", 1.0))
            xv = float(x)
            t = self._norm(xv, lo, hi)
        except (TypeError, ValueError):
            # Non-numeric domain used with numeric kind → fall back safely.
            # This prevents crashes when someone accidentally reuses a numeric kind
            # on a tuple domain. Prefer 'custom' for such domains.
            return float(min(max(self.base, 0.0), 1.0))

        # 3) Numeric kinds
        if self.kind == "constant":
            b = self.base

        elif self.kind == "linear_high":
            b = self.degree * t

        elif self.kind == "linear_low":
            b = self.degree * (1.0 - t)

        elif self.kind == "step_value":
            # threshold is in ORIGINAL units (same as x)
            b = self.degree if xv >= self.threshold else 0.0

        elif self.kind == "window":
            # window [lo, hi] in ORIGINAL units (self.lo/self.hi)
            b = self.height if (xv >= self.lo and xv <= self.hi) else 0.0

        elif self.kind == "gaussian":
            # mu/sigma are in ORIGINAL units
            b = self.degree * math.exp(- (xv - self.mu) ** 2 / (2.0 * (self.sigma ** 2) + EPS_COMPARE))

        elif self.kind == "sigmoid":
            # sigmoid in normalized space t ∈ [0,1], centered at self.center
            b = self.degree * (1.0 / (1.0 + math.exp(-self.ksig * (t - self.center))))

        elif self.kind == "piecewise":
            # piecewise-linear in normalized t; knots_t/knots_y define the shape
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
            # Unknown kind → base
            b = self.base

        return float(min(max(b, 0.0), 1.0))


@dataclass
class CompositeBias:
    """
    Combine multiple Bias1D rules using 'max' (default) or 'sum' (clamped to [0,1]).
    Useful when composing window + sigmoid, etc.
    """
    rules: List[Bias1D]
    combine: str = "max"  # 'max' or 'sum'

    def bias_for_value(self, x: float, domain_info: Dict[str, float]) -> float:
        vals = [r.bias_for_value(x, domain_info) for r in self.rules]
        if not vals:
            return 0.0
        if self.combine == "sum":
            return float(min(max(sum(vals), 0.0), 1.0))
        return float(min(max(max(vals), 0.0), 1.0))


def biases_from_bias_obj(domain: List[float], bias_obj: Any) -> Dict[float, float]:
    """Materialize a per-value bias map for a given domain and Bias1D/CompositeBias or custom-like object."""
    if not domain:
        return {}
    info = {"min": min(domain), "max": max(domain)}
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}


# Handy random bias generators (unchanged from your code)
def make_random_sparse_bias(
    domain_values: List[float], high: float = 0.9, low: float = 0.1,
    frac_hot: float = 0.3, seed: Optional[int] = 42, round_ndigits: int = 6,
) -> Bias1D:
    """Two-level random bias over DISTINCT domain."""
    rng = np.random.default_rng(seed)
    dom = sorted(set(domain_values))
    m = max(1, int(round(frac_hot * len(dom))))
    hot = set(rng.choice(dom, size=m, replace=False).tolist())
    hot_rounded = {round(x, round_ndigits) for x in hot}
    def f(value: float, _info: Dict[str, float]) -> float:
        key = round(float(value), round_ndigits)
        return high if key in hot_rounded else low
    return Bias1D(kind="custom", custom=f)


def make_random_multilevel_bias(
    domain_values: List[float],
    levels: Tuple[float, float, float, float] = (0.9, 0.6, 0.3, 0.0),  # high, med, low, none
    probs:  Tuple[float, float, float, float] = (0.2, 0.3, 0.3, 0.2),
    seed: Optional[int] = 123,
    round_ndigits: int = 6,
) -> Bias1D:
    """Four-level random bias: randomly assigns each distinct value to a level."""
    rng = np.random.default_rng(seed)
    dom = sorted(set(domain_values))
    labels = rng.choice(4, size=len(dom), p=np.array(probs) / sum(probs))
    value2bias = {round(v, round_ndigits): levels[int(lbl)] for v, lbl in zip(dom, labels)}

    def f(value: float, _info: Dict[str, float]) -> float:
        return value2bias.get(round(float(value), round_ndigits), 0.0)

    return Bias1D(kind="custom", custom=f)

# =============================================================================
# Receiver models & utilities
# =============================================================================

def system_best_response_threshold(
    q_groups: List[List[float]], domain: List[float],
    post: Dict[float, float], biases: Dict[float, float],
) -> Dict[float, int]:
    """Threshold receiver: keep(v) = 1{post[v] > bias[v]}."""
    return {
        v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0
        for g in q_groups for v in g
    }


def system_best_response_quadratic(
    q_groups: List[List[float]], domain: List[float],
    post: Dict[float, float], biases: Dict[float, float],
    gamma: float = 1.0,
) -> Dict[float, float]:
    """Quadratic receiver: action(v) = clip(gamma * post[v] + bias[v], 0, 1)."""
    out: Dict[float, float] = {}
    for g in q_groups:
        for v in g:
            a = gamma * post.get(v, 0.0) + biases.get(v, 0.0)
            out[v] = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return out


def system_best_response(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0,
):
    """Convenience wrapper that computes posteriors first and dispatches to the chosen receiver."""
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    if receiver_model == "quadratic":
        return system_best_response_quadratic(q_groups, domain, post, biases, gamma=gamma)
    raise ValueError("receiver_model must be 'threshold' or 'quadratic'")


def _exposure_weights(order_list: List[float], scheme: str = "harmonic") -> Dict[float, float]:
    """
    Generates exposure weights for a presented order to add a tiny epsilon bonus
    that prefers “nicer-looking” rankings in tie-breaks only.
    """
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
    theta: Dict[float, float], response, receiver_model: str,
    *, order_list: Optional[List[float]] = None, eps_order: float = 0.0, exposure_scheme: str = "harmonic",
) -> float:
    """
    User utility:
    - threshold: sum_v θ[v] * keep[v]
    - quadratic: -sum_v (action[v] - θ[v])^2
    + optional tiny order bonus for presentation (never changes core optimality).
    """
    if receiver_model == "threshold":
        base = sum(theta.get(v, 0.0) * a for v, a in response.items())
        act = response  # 0/1 policy
    else:
        base = -sum((response.get(v, 0.0) - theta.get(v, 0.0)) ** 2 for v in theta.keys())
        act = response  # continuous [0,1]

    if eps_order > 0.0 and order_list is not None:
        w = _exposure_weights(order_list, scheme=exposure_scheme)
        bonus = sum(w.get(v, 0.0) * theta.get(v, 0.0) * float(act.get(v, 0.0)) for v in order_list)
        return base + eps_order * bonus
    return base

# =============================================================================
# Small helpers
# =============================================================================

def flatten_in_order(q_groups: List[List[Any]]) -> List[Any]:
    """Flatten list-of-groups preserving (group, then in-group) order."""
    return [x for g in q_groups for x in g]

def op_pairs_strict(q_groups: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Return all strict ordered pairs (u,v) that appear in the query (inter-group only).
    This is exactly the set OP(q) over which Alg. 1 tests credibility.
    """
    pairs: List[Tuple[float, float]] = []
    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def swap_single_pair(q_groups: List[List[float]], u: float, v: float) -> List[List[float]]:
    """Construct the “swap query” used in Alg. 1 for a specific pair (u,v)."""
    return [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]

def dedupe_and_sort_desc(values: List[float]) -> List[float]:
    """Distinct domain in DESC order."""
    return sorted(set(values), reverse=True)

def _validate_partition_or_die(q_groups: List[List[float]], domain: List[float]) -> None:
    """
    Ensure q_groups is a partition of the domain (covers each element exactly once).
    This keeps all algorithms well-defined when starting from a partial ranking.
    """
    flat = [x for g in q_groups for x in g]
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_groups must be a partition of `domain` (cover each value exactly once, no dups).")


# =============================================================================
# Alg. 1, 2, 4
# =============================================================================

def _is_fully_symmetric_bias(biases: Dict[float, float]) -> bool:
    """Helper for tests: treat nearly-constant bias as fully symmetric."""
    if not biases:
        return True
    vals = list(biases.values())
    return (max(vals) - min(vals)) <= EPS_COMPARE


def algorithm_1_credibility_detection(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order: float = 0.0,
    tie_policy: str = "auto",          # "auto" | "neutral" | "force_on_ties"
    rule: str = "paper",               # "paper" | "strict_opposite"
) -> Set[Tuple[float, float]]:
    """
    Alg. 1: compute set C of credible strict edges present in q_groups.

    We implement a fast path for (uniform prior + threshold receiver + eps_order≈0),
    under which swapping a pair (u,v) only reassigns group-average posteriors ai/aj
    to those two items; everything else remains unchanged. This lets us compute the
    2×2 utilities via O(1) deltas instead of re-solving the whole game.

    'rule':
      - "paper": (u,v) is credible unless BOTH types strictly prefer the same column
                 (i.e., not(both_prefer_q or both_prefer_swap)).
      - "strict_opposite": (u,v) is credible iff the two types make strict & opposite
                           choices between q and swap(u,v).

    Fallback: for other priors/receivers or eps_order>0, we use the generic definition.
    """
    # Short-circuit: symmetric biases → treat all strict pairs as credible (test convenience)
    if tie_policy == "auto" and _is_fully_symmetric_bias(biases):
        return set(op_pairs_strict(q_groups))

    # Fast path conditions
    fast = (prior.kind == "uniform" and receiver_model == "threshold" and abs(eps_order) <= 1e-12)

    if fast:
        # Flatten items in presented order
        items = [x for g in q_groups for x in g]
        k = len(domain)
        idx = {v: i for i, v in enumerate(items)}

        # group id for each item (in the q order)
        gid_of = [None] * len(items)
        for gi, g in enumerate(q_groups):
            for v in g:
                gid_of[idx[v]] = gi

        # Uniform group posteriors a_g[gi] = average of the rank block of group gi
        a_g = []
        r = 1
        for g in q_groups:
            n = len(g)
            j_hi = k - r + 1
            j_lo = k - (r + n - 1) + 1
            a_g.append(((j_hi + j_lo) / 2.0) / (k + 1.0))
            r += n

        # θ⁺ for each item under q (group-average for its group)
        theta_plus = [a_g[gid_of[i]] for i in range(len(items))]

        # Threshold actions under q
        beta_q = [1 if (theta_plus[i] - biases[items[i]]) > EPS_COMPARE else 0
                  for i in range(len(items))]

        # U(θ⁺, β(q)) — base utility for the '+' type
        U_pp = sum(theta_plus[i] * beta_q[i] for i in range(len(items)))

        C: Set[Tuple[float, float]] = set()

        # Check every strict inter-group pair (u in Gi, v in Gj, i<j)
        for gi in range(len(q_groups)):
            for gj in range(gi + 1, len(q_groups)):
                ai, aj = a_g[gi], a_g[gj]
                for u in q_groups[gi]:
                    iu = idx[u]; bu = biases[u]
                    # If u moves to group j, it gets posterior aj
                    betas_u = 1 if (aj - bu) > EPS_COMPARE else 0
                    for v in q_groups[gj]:
                        iv = idx[v]; bv = biases[v]
                        # If v moves to group i, it gets posterior ai
                        betas_v = 1 if (ai - bv) > EPS_COMPARE else 0

                        # 2×2 utilities via O(1) deltas (no recomputation)
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
                        else:  # "strict_opposite"
                            cplus_q  = (up_q > up_s + EPS_COMPARE)
                            cplus_sw = (up_s > up_q + EPS_COMPARE)
                            cminus_q  = (um_q > um_s + EPS_COMPARE)
                            cminus_sw = (um_s > um_q + EPS_COMPARE)
                            if (cplus_q and cminus_sw) or (cplus_sw and cminus_q):
                                C.add((u, v))
        return C

    # --------- Generic fallback (supports other priors/receivers or eps_order>0) ----------
    C: Set[Tuple[float, float]] = set()

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
        else:  # "strict_opposite"
            cplus  = _decide(up_q, up_s)
            cminus = _decide(um_q, um_s)
            if tie_policy == "force_on_ties":
                if cplus is None:  cplus  = "q"
                if cminus is None: cminus = "swap"
            if cplus is not None and cminus is not None and cplus != cminus:
                C.add((u, v))

    return C



def _reachability_bitsets(items: List[float], edges: Set[Tuple[float, float]]):
    """Bitset transitive closure over edges restricted to item order."""
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

def _boundary_supported(idx_map, R, Gi: List[float], Gj: List[float],
                        *, policy: str = "all", alpha: float = 0.5) -> bool:
    """
    Return whether the boundary Gi -> Gj is supported by credibility reachability.

    policy:
      - "all":      every (u,v) with u∈Gi, v∈Gj is reachable (original behavior)
      - "any":      at least one (u,v) is reachable
      - "quantile": at least alpha fraction of pairs are reachable (0<alpha<=1)
    """
    if not Gi or not Gj:
        return True

    # Precompute a bitmask of Gj nodes
    want_mask = 0
    for v in Gj:
        want_mask |= (1 << idx_map[v])

    if policy == "all":
        for u in Gi:
            if (R[idx_map[u]] & want_mask) != want_mask:
                return False
        return True

    # Count supported pairs
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
    initial_order: List[float], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0,
    eps_order: float = 0.0, rule: str = "paper",
    boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[float]]:
    q_cur: List[List[float]] = [[x] for x in initial_order]
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
            if not _boundary_supported(idx_map, R, Gi, Gj,
                                       policy=boundary_policy, alpha=alpha):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur



def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[float]], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0,
    eps_order: float = 0.0, rule: str = "paper",
    boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[float]]:
    _validate_partition_or_die(q_init_groups, domain)
    q_cur = [g[:] for g in q_init_groups]
    for _ in range(max(0, len(q_cur) - 1) + 1):
        C = algorithm_1_credibility_detection(
            q_cur, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma, eps_order=0.0, rule=rule, tie_policy="neutral"
        )
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported(idx_map, R, Gi, Gj,
                                       policy=boundary_policy, alpha=alpha):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur


def algorithm_4_maximally_informative(
    q_base: List[List[float]],
    domain: List[float],
    prior: PriorSpec,
    *,
    biases: Dict[float, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order_for_tiebreak: float = 0.0
) -> List[List[float]]:
    """
    Alg. 4: DP over contiguous runs of q_base to select merges that yield
    strictly positive base utility gains (under the chosen receiver model).
    A tiny order-aware score is used ONLY to break DP ties.
    """
    m = len(q_base)
    if m == 0:
        return []

    # θ aligned with domain singletons (for consistent scoring across runs)
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    # Precompute "baseline" block utilities for q_base (both base & epsilon versions)
    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model, gamma=gamma)
    base_gain0: List[float] = []
    base_gain_eps: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        resp_t = {x: beta_base.get(x, 0 if receiver_model == "threshold" else 0.0) for x in items_t}
        # Base objective (no presentation bonus)
        u0 = user_utility_from_response(theta, resp_t, receiver_model, order_list=None, eps_order=0.0)
        # Tiny bonus used only for DP tie-breaks
        ueps = user_utility_from_response(theta, resp_t, receiver_model, order_list=items_t, eps_order=eps_order_for_tiebreak)
        base_gain0.append(u0)
        base_gain_eps.append(ueps)

    # Build marginal gain tables C0 (base) and Ceps (tie-break score)
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

    # If no strictly positive base gains, keep q_base
    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    # Standard DP with tie-breaks: prefer higher base, then higher eps, then later cuts.
    dp   = [0.0] * (m + 1)
    tie  = [0.0] * (m + 1)  # accumulated eps-score
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

    q_star: List[List[float]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(
    domain_values: List[float],
    prior: PriorSpec,
    bias_obj: Any,
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    q_user: Optional[List[List[float]]] = None,  # optional partial ranking input
) -> Dict[str, Any]:
    """
    Back-compatible pipeline wrapper:
    - If q_user is None: start from singletons (strict order).
    - If q_user is provided: honor user-provided ties and only MERGE where unsupported.
    Also returns utilities for q_babble (one big tie), q_base, and q★.
    """
    domain = dedupe_and_sort_desc(domain_values)
    biases = biases_from_bias_obj(domain, bias_obj)

    # q_babble: one big tie (no information)
    q_babble = [domain[:]]

    if q_user is None:
        # Original: start from strict order (singletons)
        q_base = algorithm_2_build_qbase(
            domain, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma, eps_order=EPS_ORDER,
            rule="paper",
            boundary_policy="quantile",  # <-- soften the boundary test
            alpha=0.5  # <-- at least half of cross pairs reachable
        )
    else:
        # New: start from the user’s partial ranking (ties intact)
        _validate_partition_or_die(q_user, domain)
        q_base = algorithm_2_build_qbase_from_groups(
            q_user, domain, prior,
            biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=EPS_ORDER
        )

    q_star = algorithm_4_maximally_informative(
        q_base, domain, prior, biases=biases,
        receiver_model=receiver_model, gamma=gamma, eps_order_for_tiebreak=EPS_ORDER
    )

    # ---------- Utilities (no presentation bonus) ----------
    util_babble, kept_babble = evaluate_plan_utility(
        q_babble, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma
    )
    util_base, kept_base = evaluate_plan_utility(
        q_base, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma
    )
    util_star, kept_star = evaluate_plan_utility(
        q_star, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma
    )

    out = {
        "domain": domain,
        "q_user": q_user,
        "q_babble": q_babble,
        "q_base": q_base,
        "q_star": q_star,
        "biases": biases,

        # New utility report
        "utility": {
            "q_babble": util_babble,
            "q_base":   util_base,
            "q_star":   util_star,
        },
    }

    # For threshold receiver, also include #kept
    if receiver_model == "threshold":
        out["kept_counts"] = {
            "q_babble": kept_babble,
            "q_base":   kept_base,
            "q_star":   kept_star,
        }

    return out


def run_pipeline_domain(
    domain_values: List[float], prior: PriorSpec, bias_obj: Any,
    receiver_model: str = "threshold", gamma: float = 1.0
) -> Dict[str, Any]:
    """
    Legacy entry-point preserved for existing callers.
    Equivalent to run_pipeline(..., q_user=None).
    """
    return run_pipeline(domain_values, prior, bias_obj, receiver_model=receiver_model, gamma=gamma, q_user=None)

# =============================================================================
# Reporting helpers (unchanged)
# =============================================================================

def report_diff_to_qbase(q_base: List[List[float]]):
    print("\n--- Change Summary (Input -> q_base) ---")
    merged_groups = [group for group in q_base if len(group) > 1]
    if not merged_groups:
        print("  - No changes. All strict boundaries were credible.")
    else:
        print(f"  - {len(merged_groups)} merge(s) due to non-credible boundaries:")
        for i, group in enumerate(merged_groups, 1):
            print(f"    - Merge {i}: {sorted(group, reverse=True)}")
    print("  -----------------------------------------")


def report_diff_to_qstar(q_base: List[List[float]], q_star: List[List[float]]):
    print("\n--- Change Summary (q_base -> q_star) ---")
    if q_base == q_star:
        print("  - No changes. q_base is already maximally informative.")
        print("  -----------------------------------------")
        return
    q_it = iter(q_base)
    for g in q_star:
        packed, size = [], 0
        while size < len(g):
            gb = next(q_it)
            packed.append(gb)
            size += len(gb)
        if len(packed) > 1:
            print(f"  - Merged {packed} into {g}")
    print("  (Merges chosen to maximize user utility under the chosen receiver)")
    print("  -----------------------------------------")


# ============================================================================
# Exhaustic Search Experiment
# ============================================================================
# === Additions for the grid experiment =======================================
# === Posterior-driven bias-level grid over domain size n (logs q_babble/q_base/q_star) ===
import os, time, json, itertools, csv as _csv
import numpy as np

def _nonincreasing_level_tuples(cands: List[float], m: int = 4):
    """Yield nonincreasing tuples (ℓ1 >= ℓ2 >= ... >= ℓm) from candidate scalars."""
    cs = sorted(set(float(x) for x in cands), reverse=True)
    for tpl in itertools.combinations_with_replacement(cs, m):
        yield tpl

def posterior_candidate_levels(n: int, *, count: int = 10, include_extremes: bool = True) -> List[float]:
    """
    Candidate levels taken FROM the posterior grid {j/(n+1), j=1..n}.
    We sub-sample ~`count` quantiles to keep grids manageable.
    """
    if n <= 0:
        return []
    idxs = np.linspace(1, n, num=max(1, count))
    idxs = sorted({int(round(x)) for x in idxs})
    levels = [i / (n + 1.0) for i in idxs]
    if include_extremes:
        levels.extend([1.0/(n+1.0), n/(n+1.0)])
    return sorted(set(levels), reverse=True)

def algorithm_2_build_qbase_p(
    initial_order: List[float], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0,
    rule: str = "paper", tie_policy: str = "neutral",
    boundary_policy: str = "all", alpha: float = 0.5,
) -> List[List[float]]:
    """
    Alg.2 with explicit tie & boundary policies.
    """
    q_cur: List[List[float]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(
            q_cur, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma,
            eps_order=0.0, rule=rule, tie_policy=tie_policy
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

def run_bias_grid_experiment_from_posteriors(
    *,
    n_min: int = 2,
    n_max: int = 1000,
    posterior_level_count: int = 10,      # how many posterior grid points per n
    probs: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    out_csv: str = "../results/tables/bias_grid_from_posteriors_full.csv",
    rule: str = "paper",
    tie_policy: str = "neutral",          # avoid 'auto' short-circuit
    boundary_policy: str = "all",
    alpha: float = 0.5,
    include_star: bool = True,            # also run Alg.4 and log q★
    print_progress: bool = True,
):
    """
    For each domain size n in [n_min..n_max]:
      - domain = {n, n-1, ..., 1}
      - candidate levels = subsampled posteriors {j/(n+1)} (DESC)
      - grid over 4-level nonincreasing tuples (ℓ1>=ℓ2>=ℓ3>=ℓ4)
      - biases = random multilevel using those levels (seeded by (n, levels))
      - compute:
          * q_babble (one big tie), q_base (Alg.2), q_star (Alg.4, optional)
          * utilities U(q_babble), U(q_base), U(q_star)
          * deltas vs babble: Δ_base, Δ_star
          * kept counts (threshold)
          * Alg.1-first |C| and time; Alg.2 time; Alg.4 time; (2+4) combined time
      - keep the levels tuple that maximizes Δ_base
      - write ONE best row per n to CSV
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    prior = PriorSpec(kind="uniform")
    want_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        w = _csv.writer(f, quoting=_csv.QUOTE_NONNUMERIC)
        if want_header:
            w.writerow([
                "n", "candidate_levels_count", "best_levels_json", "probs_json",
                "utility_q_babble", "utility_q_base", "utility_q_star",
                "delta_base", "delta_star",
                "kept_q_babble", "kept_q_base", "kept_q_star",
                "q_base_groups", "q_star_groups", "merges_alg2",
                "num_edges_alg1_first",
                "time_alg1_first_s", "time_alg2_s", "time_alg4_s", "time_alg24_s"
            ])
            f.flush()

        for n in range(n_min, n_max + 1):
            domain = list(range(n, 0, -1))  # DESC distinct domain
            cand_levels = posterior_candidate_levels(n, count=posterior_level_count, include_extremes=True)
            level_tuples = list(_nonincreasing_level_tuples(cand_levels, 4))

            best = None  # (Δ_base, row_dict)
            for levels in level_tuples:
                # deterministic seed by (n, levels)
                seed = (n * 10007 + sum(int(round(1e6 * x)) for x in levels)) & 0xFFFFFFFF

                bias_obj = make_random_multilevel_bias(
                    domain_values=domain,
                    levels=levels,    # levels FROM POSTERIORS
                    probs=probs,
                    seed=seed
                )
                biases = biases_from_bias_obj(domain, bias_obj)

                # ---- Alg.1 first pass (singleton query) ----
                q_singletons = [[v] for v in domain]
                t1a = time.perf_counter()
                C0 = algorithm_1_credibility_detection(
                    q_singletons, domain, prior,
                    biases=biases, receiver_model="threshold",
                    gamma=1.0, eps_order=0.0,
                    rule=rule, tie_policy=tie_policy
                )
                t1b = time.perf_counter()
                time_alg1_first = t1b - t1a
                num_edges_alg1 = len(C0)

                # ---- Alg.2 ----
                t2a = time.perf_counter()
                q_base = algorithm_2_build_qbase_p(
                    initial_order=domain, domain=domain, prior=prior,
                    biases=biases, receiver_model="threshold", gamma=1.0,
                    rule=rule, tie_policy=tie_policy,
                    boundary_policy='all'
                )
                t2b = time.perf_counter()
                time_alg2 = t2b - t2a

                # ---- Utilities (+ kept) ----
                util_babble, kept_babble = evaluate_plan_utility([domain[:]], domain, prior, biases=biases, receiver_model="threshold", gamma=1.0)
                util_base,   kept_base   = evaluate_plan_utility(q_base,      domain, prior, biases=biases, receiver_model="threshold", gamma=1.0)

                # ---- Alg.4 (optional) ----
                time_alg4 = 0.0
                util_star = util_base
                kept_star = kept_base
                q_star = q_base
                if include_star:
                    t4a = time.perf_counter()
                    q_star = algorithm_4_maximally_informative(
                        q_base, domain, prior,
                        biases=biases, receiver_model="threshold", gamma=1.0,
                        eps_order_for_tiebreak=0.0
                    )
                    t4b = time.perf_counter()
                    time_alg4 = t4b - t4a
                    util_star, kept_star = evaluate_plan_utility(q_star, domain, prior, biases=biases, receiver_model="threshold", gamma=1.0)

                delta_base = float(util_base - util_babble)
                delta_star = float(util_star - util_babble)
                time_alg24 = time_alg2 + time_alg4

                row = {
                    "n": n,
                    "cand_cnt": len(cand_levels),
                    "levels": levels,
                    "probs": probs,
                    "util_babble": float(util_babble),
                    "util_base": float(util_base),
                    "util_star": float(util_star),
                    "delta_base": delta_base,
                    "delta_star": delta_star,
                    "kept_babble": kept_babble,
                    "kept_base": kept_base,
                    "kept_star": kept_star,
                    "qbase_groups": len(q_base),
                    "qstar_groups": len(q_star),
                    "merges_alg2": n - len(q_base),
                    "num_edges_alg1": num_edges_alg1,
                    "t1": time_alg1_first,
                    "t2": time_alg2,
                    "t4": time_alg4,
                    "t24": time_alg24,
                }
                if (best is None) or (delta_base > best[0] + EPS_COMPARE):
                    best = (delta_base, row)

            br = best[1]
            w.writerow([
                br["n"], br["cand_cnt"], json.dumps(tuple(map(float, br["levels"]))), json.dumps(tuple(map(float, br["probs"]))),
                br["util_babble"], br["util_base"], br["util_star"],
                br["delta_base"], br["delta_star"],
                br["kept_babble"], br["kept_base"], br["kept_star"],
                br["qbase_groups"], br["qstar_groups"], br["merges_alg2"],
                br["num_edges_alg1"],
                br["t1"], br["t2"], br["t4"], br["t24"],
            ])
            f.flush()

            if print_progress and (n <= 10 or n % 25 == 0):
                print(f"[n={n:4d}] Δ_base={br['delta_base']:.6f}, Δ_star={br['delta_star']:.6f} | "
                      f"levels={br['levels']} | q_base={br['qbase_groups']}→q★={br['qstar_groups']} | "
                      f"t1={br['t1']:.4f}s t2={br['t2']:.4f}s t4={br['t4']:.4f}s (2+4={br['t24']:.4f}s)")
    if print_progress:
        print(f"Done. CSV -> {out_csv}")
# === end experiment block ===

# === End additions ============================================================

if __name__ == "__main__":
    # Grid search over domain sizes (2..1000), bias levels drawn from POSTERIORS.
    run_bias_grid_experiment_from_posteriors(
        n_min=2,
        n_max=1000,
        posterior_level_count=10,   # raise for denser level grids (combinatorial growth!)
        probs=(0.25, 0.25, 0.25, 0.25),
        out_csv="../results/tables/bias_grid_from_posteriors_full.csv",
        rule="paper",
        tie_policy="neutral",
        boundary_policy="all",      # or "quantile" with alpha<1.0 to be less strict
        alpha=0.5,
        include_star=True,
        print_progress=True,
    )


# =============================================================================
# Example CLI (kept simple): strict input only, for smoke test
# =============================================================================
# if __name__ == "__main__":
#     DATASET_PATH = '../data/real/census.csv'
#     ORDER_BY_ATTR = 'age'
#
#     try:
#         df = pd.read_csv(DATASET_PATH)
#         df[ORDER_BY_ATTR] = pd.to_numeric(df[ORDER_BY_ATTR], errors='coerce')
#         df = df.dropna(subset=[ORDER_BY_ATTR])
#
#         #synthetic df with only 5 distinct values
#         #df = pd.DataFrame({ORDER_BY_ATTR: [20, 30, 40, 50, 60] * 20})
#
#         domain_values = df[ORDER_BY_ATTR].unique().tolist()
#
#         #get posteriors for uniform prior for the domain values
#         posteriors=compute_expected_posteriors([[v] for v in domain_values], domain_values, PriorSpec(kind="uniform"))
#         min_posterior = min(posteriors.values())#0.0133
#         max_posterior = max(posteriors.values())#0.9866
#
#
#         prior = PriorSpec(kind="uniform")
#         bias  = make_random_multilevel_bias(domain_values, levels=(0.5, 0.6, 0.2, 0.2), probs=(0.25, 0.25, 0.25, 0.25), seed=123)
#
#         out_thr = run_pipeline(domain_values, prior, bias, receiver_model="threshold", gamma=1.0, q_user=None)
#
#         fmt = lambda q: " ≺ ".join("{" + ", ".join(map(lambda x: f"{x:.4g}", g)) + "}" for g in q)
#         print("\n[THRESHOLD]")
#         print("Distinct domain:", [f"{x:.4g}" for x in out_thr["domain"]])
#         print("q_base:", fmt(out_thr["q_base"]))
#         print("q★:    ", fmt(out_thr["q_star"]))
#         report_diff_to_qbase(out_thr["q_base"])
#         report_diff_to_qstar(out_thr["q_base"], out_thr["q_star"])
#         print("Utility(q_babble) =", out_thr["utility"]["q_babble"])
#         print("Utility(q_base)   =", out_thr["utility"]["q_base"])
#         print("Utility(q★)       =", out_thr["utility"]["q_star"])
#         if "kept_counts" in out_thr:
#             print("Kept(q_babble) =", out_thr["kept_counts"]["q_babble"])
#             print("Kept(q_base)   =", out_thr["kept_counts"]["q_base"])
#             print("Kept(q★)       =", out_thr["kept_counts"]["q_star"])
#
#     except FileNotFoundError:
#         print(f"Dataset not found at '{DATASET_PATH}'.")
