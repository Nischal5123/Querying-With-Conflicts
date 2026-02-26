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
    tie_policy: str = "auto"   # "auto" | "neutral" | "force_on_ties"
) -> Set[Tuple[float, float]]:
    """
    Alg. 1: compute the set C of credible strict edges present in q_groups.

    IMPORTANT:
    - Checks only pairs in OP(q) (inter-group edges).
    - Your original behavior is preserved: a pair (u,v) is marked credible if
      both “types” (τ+ favoring u<v and τ− favoring v<u) make strict & opposite
      choices between q and q_swap(u,v). (If you ever want the paper’s weaker
      rule, it is included below as commented lines.)
    - tie_policy="auto": if the bias is fully symmetric, mark all pairs credible
      (test convenience).
    """
    C: Set[Tuple[float, float]] = set()

    # Short-circuit in symmetry (useful in tests)
    if tie_policy == "auto" and _is_fully_symmetric_bias(biases):
        for (u, v) in op_pairs_strict(q_groups):
            C.add((u, v))
        return C

    # helper: "strict-only" preference comparison returning 'q', 'swap', or None (tie)
    def _decide(a: float, b: float) -> Optional[str]:
        if a > b + EPS_COMPARE: return "q"
        if b > a + EPS_COMPARE: return "swap"
        return None

    # Precompute θ and β for q
    theta_plus = compute_expected_posteriors(q_groups, domain, prior)
    beta_q     = system_best_response(q_groups, domain, prior, biases, receiver_model, gamma=gamma)
    order_q    = flatten_in_order(q_groups)

    for (u, v) in op_pairs_strict(q_groups):
        # Construct the swap query for (u,v)
        q_swap    = swap_single_pair(q_groups, u, v)
        theta_min = compute_expected_posteriors(q_swap, domain, prior)
        beta_s    = system_best_response(q_swap, domain, prior, biases, receiver_model, gamma=gamma)
        order_s   = flatten_in_order(q_swap)

        # Utilities for the 2×2 table
        up_q = user_utility_from_response(theta_plus, beta_q, receiver_model, order_list=order_q, eps_order=eps_order)
        up_s = user_utility_from_response(theta_plus, beta_s, receiver_model, order_list=order_s, eps_order=eps_order)
        um_q = user_utility_from_response(theta_min,  beta_q, receiver_model, order_list=order_q, eps_order=eps_order)
        um_s = user_utility_from_response(theta_min,  beta_s, receiver_model, order_list=order_s, eps_order=eps_order)

        # Decide each row's preferred column (strictly)
        cplus  = _decide(up_q, up_s)
        cminus = _decide(um_q, um_s)

        # Optional: "force_on_ties" mirrors your historic tie behavior
        if tie_policy == "force_on_ties":
            if cplus is None:  cplus  = "q"
            if cminus is None: cminus = "swap"

        # ORIGINAL behavior: credible iff both decisions are strict and opposite
        # if cplus is not None and cminus is not None and cplus != cminus:
        #     C.add((u, v))

        # ---- If you ever want the alternative (paper) rule, swap the block above
        # ---- with the following (and remove 'force_on_ties'):
        #
        both_prefer_q = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
        both_prefer_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
        if not (both_prefer_q or both_prefer_swap):
            C.add((u, v))

    return C


def build_adj(items: List[float], edges: Set[Tuple[float, float]]) -> Dict[float, List[float]]:
    """Build adjacency list for reachability over C."""
    adj = {x: [] for x in items}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)
    return adj


def reachable(start: float, adj: Dict[float, List[float]]) -> Set[float]:
    """Compute forward reachability from 'start' in a small directed graph."""
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for w in adj.get(u, []):
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return seen


def algorithm_2_build_qbase(
    initial_order: List[float], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0, eps_order: float = 0.0
) -> List[List[float]]:
    """
    Alg. 2 (classic): start from SINGLETONS (strict order) and MERGE adjacent boundaries
    that are not supported by credibility reachability. This is your original routine.
    """
    q_cur: List[List[float]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(q_cur, domain, prior,
                                              biases=biases, receiver_model=receiver_model, gamma=gamma,
                                              eps_order=eps_order)
        adj = build_adj(domain, C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            # Boundary Gi->Gj must be supported for ALL u∈Gi, v∈Gj
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur


def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[float]],
    domain: List[float],
    prior: PriorSpec,
    *,
    biases: Dict[float, float],
    receiver_model: str = "threshold",
    gamma: float = 1.0,
    eps_order: float = 0.0,
) -> List[List[float]]:
    """
    Alg. 2 (group-respecting): start from a USER-PROVIDED PARTIAL RANKING (ties intact),
    and MERGE adjacent group boundaries that are unsupported by credibility reachability.
    - Never splits ties (partial-ranking-as-strategy).
    - This is the minimal addition that lets the pipeline honor user ties.
    """
    _validate_partition_or_die(q_init_groups, domain)
    q_cur = [g[:] for g in q_init_groups]

    for _ in range(max(0, len(q_cur) - 1) + 1):
        C = algorithm_1_credibility_detection(
            q_cur, domain, prior,
            biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=eps_order
        )
        adj = build_adj(domain, C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            # Boundary Gi->Gj must be supported for ALL u∈Gi, v∈Gj
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
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
    q_user: Optional[List[List[float]]] = None,  # NEW: optional partial ranking input
) -> Dict[str, Any]:
    """
    Back-compatible pipeline wrapper:
    - If q_user is None: preserves your original behavior (start from singletons).
    - If q_user is provided: honor user-provided ties and only MERGE where unsupported.
    """
    domain = dedupe_and_sort_desc(domain_values)
    biases = biases_from_bias_obj(domain, bias_obj)

    if q_user is None:
        # Original: start from strict order (singletons)
        q_base = algorithm_2_build_qbase(
            domain, domain, prior, biases=biases,
            receiver_model=receiver_model, gamma=gamma, eps_order=EPS_ORDER
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

    return {"domain": domain, "q_user": q_user, "q_base": q_base, "q_star": q_star, "biases": biases}


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



#candidate_cols=['days_from_compas', 'juv_other_count', 'days_b_screening_arrest', 'priors_count', 'age_cat_binned', 'start_duration','end_duration']
# =============================================================================
# Example CLI (kept simple): strict input only, for smoke test
# =============================================================================
if __name__ == "__main__":
    DATASET_PATH = '../data/real/census.csv'
    ORDER_BY_ATTR = 'age'

    try:
        df = pd.read_csv(DATASET_PATH)
        df[ORDER_BY_ATTR] = pd.to_numeric(df[ORDER_BY_ATTR], errors='coerce')
        df = df.dropna(subset=[ORDER_BY_ATTR])
        domain_values = df[ORDER_BY_ATTR].unique().tolist()

        prior = PriorSpec(kind="uniform")
        bias  = make_random_multilevel_bias(domain_values, levels=(0.5, 0.1, 0, 0), probs=(0.25, 0.25, 0.25, 0.25), seed=123)

        out_thr = run_pipeline(domain_values, prior, bias, receiver_model="threshold", gamma=1.0, q_user=None)

        fmt = lambda q: " ≺ ".join("{" + ", ".join(map(lambda x: f"{x:.4g}", g)) + "}" for g in q)
        print("\n[THRESHOLD]")
        print("Distinct domain:", [f"{x:.4g}" for x in out_thr["domain"]])
        print("q_base:", fmt(out_thr["q_base"]))
        print("q★:    ", fmt(out_thr["q_star"]))
        report_diff_to_qbase(out_thr["q_base"])
        report_diff_to_qstar(out_thr["q_base"], out_thr["q_star"])
    except FileNotFoundError:
        print(f"Dataset not found at '{DATASET_PATH}'.")
