# -*- coding: utf-8 -*-
"""
Domain-distinct COI analysis with threshold & quadratic receivers + tests.

Key points:
- DISTINCT domain (ties collapsed)
- Bias over numeric domain only (no frequency)
- Priors → expected posteriors by rank; groups average those ranks
- Receivers:
    threshold: keep(v) = 1{post[v] > bias[v]}
    quadratic: action(v) = clip(gamma*post[v] + bias[v], 0, 1)
- Utility (intent-aware): base utility + tiny order-aware bonus
- Alg. 1: (u,v) credible if τ+ and τ- disagree after EPS comparisons
- Alg. 2: merge adjacent boundaries not supported by C-reachability
- Alg. 4: marginal DP with no-positive-gain safeguard and tie-break to fewer merges
- Tests (focus on threshold): symmetry, credible set size, base/star under symmetry,
  and bias-induced merges.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
from collections import deque
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Global epsilons
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-12   # numeric comparisons tolerance
EPS_ORDER   = 1e-9    # tiny bonus for order-aware tie-break (utility)

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
        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec
) -> Dict[float, float]:
    """Average rank expectations within each group."""
    k = len(domain)
    post: Dict[float, float] = {}
    r = 1
    for g in q_groups:
        n = len(g)
        if n == 0: continue
        a = sum(prior.rank_expectation(k, rr) for rr in range(r, r + n)) / float(n)
        for v in g: post[v] = a
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
    custom: Optional[Callable[[float, Dict[str, float]], float]] = None

    def _norm(self, x: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    def bias_for_value(self, x: float, domain_info: Dict[str, float]) -> float:
        lo = float(domain_info["min"]); hi = float(domain_info["max"])
        t = self._norm(float(x), lo, hi)

        if self.kind == "constant":
            b = self.base
        elif self.kind == "linear_high":
            b = self.degree * t
        elif self.kind == "linear_low":
            b = self.degree * (1.0 - t)
        elif self.kind == "step_value":
            b = self.degree if float(x) >= self.threshold else 0.0
        elif self.kind == "window":
            b = self.height if (float(x) >= self.lo and float(x) <= self.hi) else 0.0
        elif self.kind == "gaussian":
            b = self.degree * math.exp(- (float(x) - self.mu) ** 2 / (2.0 * (self.sigma ** 2) + EPS_COMPARE))
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
        elif self.kind == "custom" and self.custom:
            b = float(self.custom(float(x), domain_info))
        else:
            b = self.base

        return float(min(max(b, 0.0), 1.0))


@dataclass
class CompositeBias:
    rules: List[Bias1D]
    combine: str = "max"  # 'max' or 'sum'

    def bias_for_value(self, x: float, domain_info: Dict[str, float]) -> float:
        vals = [r.bias_for_value(x, domain_info) for r in self.rules]
        if not vals: return 0.0
        if self.combine == "sum":
            return float(min(max(sum(vals), 0.0), 1.0))
        return float(min(max(max(vals), 0.0), 1.0))


def biases_from_bias_obj(domain: List[float], bias_obj: Any) -> Dict[float, float]:
    if not domain: return {}
    info = {"min": min(domain), "max": max(domain)}
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}


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
    """
    Four-level random bias: assigns each distinct value to {high, med, low, none}
    with given probabilities and returns a custom Bias1D.
    """
    rng = np.random.default_rng(seed)
    dom = sorted(set(domain_values))
    labels = rng.choice(4, size=len(dom), p=np.array(probs) / sum(probs))
    value2bias = {round(v, round_ndigits): levels[int(lbl)] for v, lbl in zip(dom, labels)}

    def f(value: float, _info: Dict[str, float]) -> float:
        return value2bias.get(round(float(value), round_ndigits), 0.0)

    return Bias1D(kind="custom", custom=f)

# =============================================================================
# Receiver models & utilities (order-aware)
# =============================================================================

def system_best_response_threshold(
    q_groups: List[List[float]], domain: List[float], post: Dict[float, float], biases: Dict[float, float],
) -> Dict[float, int]:
    return {v: 1 if post.get(v, 0.0) > biases.get(v, 0.0) else 0 for g in q_groups for v in g}


def system_best_response_quadratic(
    q_groups: List[List[float]], domain: List[float], post: Dict[float, float], biases: Dict[float, float], gamma: float = 1.0,
) -> Dict[float, float]:
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
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    if receiver_model == "quadratic":
        return system_best_response_quadratic(q_groups, domain, post, biases, gamma=gamma)
    raise ValueError("receiver_model must be 'threshold' or 'quadratic'")


def _exposure_weights(order_list: List[float], scheme: str = "harmonic") -> Dict[float, float]:
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
    theta: Dict[float, float], response, receiver_model: str,
    *, order_list: Optional[List[float]] = None, eps_order: float = 0.0, exposure_scheme: str = "harmonic",
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
# Helpers
# =============================================================================

def flatten_in_order(q_groups: List[List[Any]]) -> List[Any]:
    return [x for g in q_groups for x in g]

def op_pairs_strict(q_groups: List[List[float]]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def swap_single_pair(q_groups: List[List[float]], u: float, v: float) -> List[List[float]]:
    return [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]

def dedupe_and_sort_desc(values: List[float]) -> List[float]:
    return sorted(set(values), reverse=True)

# =============================================================================
# Alg. 1, 2, 4
# =============================================================================

def algorithm_1_credibility_detection(
    q_groups: List[List[float]], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0,
) -> Set[Tuple[float, float]]:
    C: Set[Tuple[float, float]] = set()
    theta_plus = compute_expected_posteriors(q_groups, domain, prior)
    beta_q = system_best_response(q_groups, domain, prior, biases, receiver_model, gamma=gamma)
    order_q = flatten_in_order(q_groups)

    for (u, v) in op_pairs_strict(q_groups):
        q_swap = swap_single_pair(q_groups, u, v)
        theta_minus = compute_expected_posteriors(q_swap, domain, prior)
        beta_s = system_best_response(q_swap, domain, prior, biases, receiver_model, gamma=gamma)
        order_s = flatten_in_order(q_swap)

        up_q = user_utility_from_response(theta_plus, beta_q, receiver_model, order_list=order_q, eps_order=EPS_ORDER)
        up_s = user_utility_from_response(theta_plus, beta_s, receiver_model, order_list=order_s, eps_order=EPS_ORDER)
        um_q = user_utility_from_response(theta_minus, beta_q, receiver_model, order_list=order_q, eps_order=EPS_ORDER)
        um_s = user_utility_from_response(theta_minus, beta_s, receiver_model, order_list=order_s, eps_order=EPS_ORDER)

        if up_q > up_s + EPS_COMPARE: cplus = "q"
        elif up_s > up_q + EPS_COMPARE: cplus = "swap"
        else: cplus = "q"          # τ+ tie → q

        if um_q > um_s + EPS_COMPARE: cminus = "q"
        elif um_s > um_q + EPS_COMPARE: cminus = "swap"
        else: cminus = "swap"       # τ- tie → swap

        if cplus != cminus:
            C.add((u, v))
    return C


def build_adj(items: List[float], edges: Set[Tuple[float, float]]) -> Dict[float, List[float]]:
    adj = {x: [] for x in items}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)
    return adj


def reachable(start: float, adj: Dict[float, List[float]]) -> Set[float]:
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
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0
) -> List[List[float]]:
    q_cur: List[List[float]] = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases,
                                              receiver_model=receiver_model, gamma=gamma)
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
    q_base: List[List[float]], domain: List[float], prior: PriorSpec, *,
    biases: Dict[float, float], receiver_model: str = "threshold", gamma: float = 1.0
) -> List[List[float]]:
    m = len(q_base)
    if m == 0: return []
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)

    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model, gamma=gamma)
    base_gain: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        if receiver_model == "threshold":
            resp_t = {x: beta_base.get(x, 0) for x in items_t}
        else:
            resp_t = {x: beta_base.get(x, 0.0) for x in items_t}
        order_t = items_t
        base_gain.append(user_utility_from_response(theta, resp_t, receiver_model,
                                                    order_list=order_t, eps_order=EPS_ORDER))

    Cmat = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            run_items = flatten_in_order(q_base[i:j + 1])
            temp_q = q_base[:i] + [run_items] + q_base[j + 1:]
            beta_run = system_best_response(temp_q, domain, prior, biases, receiver_model, gamma=gamma)
            if receiver_model == "threshold":
                resp_run = {x: beta_run.get(x, 0) for x in run_items}
            else:
                resp_run = {x: beta_run.get(x, 0.0) for x in run_items}
            run_gain = user_utility_from_response(theta, resp_run, receiver_model,
                                                  order_list=run_items, eps_order=EPS_ORDER)
            baseline = sum(base_gain[t] for t in range(i, j + 1))
            Cmat[i][j] = run_gain - baseline

    # No-positive-gain safeguard → keep q_base
    max_gain = max(Cmat[i][j] for i in range(m) for j in range(i, m))
    if max_gain <= EPS_COMPARE:
        return q_base

    # DP: only add strictly positive gains; tie-break prefers *later* cut (less merging)
    dp = [0.0] * (m + 1)
    prev = [-1] * (m + 1)
    for t in range(1, m + 1):
        best, arg = -float("inf"), -1
        for i in range(1, t + 1):
            gain = Cmat[i - 1][t - 1]
            val = dp[i - 1] + (gain if gain > EPS_COMPARE else 0.0)
            # tie on val → prefer larger (i-1) ⇒ later cut ⇒ fewer merges
            if (val > best + EPS_COMPARE) or (abs(val - best) <= EPS_COMPARE and (i - 1) > arg):
                best, arg = val, i - 1
        dp[t], prev[t] = best, arg

    q_star: List[List[float]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, flatten_in_order(q_base[i:t]))
        t = i
    return q_star

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline_domain(
    domain_values: List[float], prior: PriorSpec, bias_obj: Any,
    receiver_model: str = "threshold", gamma: float = 1.0
) -> Dict[str, Any]:
    domain = dedupe_and_sort_desc(domain_values)
    biases = biases_from_bias_obj(domain, bias_obj)
    q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases,
                                     receiver_model=receiver_model, gamma=gamma)
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases,
                                               receiver_model=receiver_model, gamma=gamma)
    return {"domain": domain, "q_base": q_base, "q_star": q_star, "biases": biases}

# =============================================================================
# Reporting
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

# =============================================================================
# Example CLI
# =============================================================================

if __name__ == "__main__":

    DATASET_PATH = '../data/real/pricerunner_aggregate.csv'
    ORDER_BY_ATTR = 'MerchantID'
    # #analyze unique values in the dataset for all columns
    # print("Analyzing dataset for unique values in each column...")
    # df = pd.read_csv(DATASET_PATH)
    # for col in df.columns:
    #     unique_vals = df[col].nunique()
    #     print(f"Column '{col}' has {unique_vals} unique values.")
    # #distinct clusterid per categoryid
    # df[ORDER_BY_ATTR] = pd.to_numeric(df[ORDER_BY_ATTR], errors='coerce')
    # df = df.dropna(subset=[ORDER_BY_ATTR])
    # distinct_per_cat = df.groupby(ORDER_BY_ATTR)['MerchantID'].nunique()
    # print("\nDistinct MerchantID counts per CategoryID and corresponsing CategoryLabels:")
    # for cat_id, count in distinct_per_cat.items():
    #     labels = df[df[ORDER_BY_ATTR] == cat_id]['CategoryLabel'].unique()
    #     print(f"CategoryID {int(cat_id)} ({', '.join(labels)}): {count} distinct MerchantID")




    try:
        df = pd.read_csv(DATASET_PATH)
        df[ORDER_BY_ATTR] = pd.to_numeric(df[ORDER_BY_ATTR], errors='coerce')
        df = df.dropna(subset=[ORDER_BY_ATTR])
        domain_values = df[ORDER_BY_ATTR].unique().tolist()
        prior = PriorSpec(kind="uniform")
        bias = make_random_multilevel_bias(domain_values, levels=(0.8, 0.9, 0.9, 0.99), probs=(0.25, 0.25, 0.25, 0.25), seed=123)
        out_thr = run_pipeline_domain(domain_values, prior, bias, receiver_model="threshold", gamma=1.0)
        fmt = lambda q: " ≺ ".join("{" + ", ".join(map(lambda x: f"{x:.4g}", g)) + "}" for g in q)
        print("\n[THRESHOLD]")
        print("Distinct domain:", [f"{x:.4g}" for x in out_thr["domain"]])
        print("q_base:", fmt(out_thr["q_base"]))
        print("q★:    ", fmt(out_thr["q_star"]))
        report_diff_to_qbase(out_thr["q_base"])
        report_diff_to_qstar(out_thr["q_base"], out_thr["q_star"])
    except FileNotFoundError:
        print(f"Dataset not found at '{DATASET_PATH}'.")