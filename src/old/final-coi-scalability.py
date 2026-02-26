
# -*- coding: utf-8 -*-
"""
COI pipeline with optional "bucket mode" (quantile binning), matching the paper's Alg. 1 rule by default.
Includes helper APIs for evaluation and a quantile-bucket workflow.

Author notes:
- The ε presentation bonus is used only for DP tie-breaking in Alg.4 (tiny).
- Alg.1 comparisons do not include ε (paper-style 2×2 test).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
from collections import deque
import math
import numpy as np

# ---------------------------------------------------------------------
# Global tolerances
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
        j = k - r_desc + 1
        return j / (k + 1.0)


def compute_expected_posteriors(q_groups: List[List[float]], domain: List[float], prior: PriorSpec) -> Dict[float, float]:
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
# Bias containers
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

    def bias_for_value(self, x, domain_info: Dict[str, float]) -> float:
        if self.kind == "custom" and self.custom:
            b = float(self.custom(x, domain_info))
            return float(min(max(b, 0.0), 1.0))
        try:
            lo = float(domain_info.get("min", 0.0)); hi = float(domain_info.get("max", 1.0)); xv = float(x)
            t = self._norm(xv, lo, hi)
        except (TypeError, ValueError):
            return float(min(max(self.base, 0.0), 1.0))
        if self.kind == "constant": b = self.base
        elif self.kind == "linear_high": b = self.degree * t
        elif self.kind == "linear_low": b = self.degree * (1.0 - t)
        elif self.kind == "step_value": b = self.degree if xv >= self.threshold else 0.0
        elif self.kind == "window": b = self.height if (xv >= self.lo and xv <= self.hi) else 0.0
        elif self.kind == "gaussian": b = self.degree * math.exp(- (xv - self.mu) ** 2 / (2.0 * (self.sigma ** 2) + EPS_COMPARE))
        elif self.kind == "sigmoid": b = self.degree * (1.0 / (1.0 + math.exp(-self.ksig * (t - self.center))))
        elif self.kind == "piecewise":
            if not self.knots_t or not self.knots_y or len(self.knots_t) != len(self.knots_y): b = self.base
            else:
                xs, ys = self.knots_t, self.knots_y
                if t <= xs[0]: b = ys[0]
                elif t >= xs[-1]: b = ys[-1]
                else:
                    i = np.searchsorted(xs, t) - 1
                    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
                    w = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
                    b = (1 - w) * y0 + w * y1
        else: b = self.base
        return float(min(max(b, 0.0), 1.0))


@dataclass
class CompositeBias:
    rules: List[Bias1D]
    combine: str = "max"

    def bias_for_value(self, x: float, domain_info: Dict[str, float]) -> float:
        vals = [r.bias_for_value(x, domain_info) for r in self.rules]
        if not vals: return 0.0
        if self.combine == "sum": return float(min(max(sum(vals), 0.0), 1.0))
        return float(min(max(max(vals), 0.0), 1.0))


def biases_from_bias_obj(domain: List[float], bias_obj: Any) -> Dict[float, float]:
    if not domain: return {}
    info = {"min": min(domain), "max": max(domain)}
    return {v: float(min(max(getattr(bias_obj, "bias_for_value")(v, info), 0.0), 1.0)) for v in domain}

# =============================================================================
# Receiver & utilities
# =============================================================================

def system_best_response_threshold(q_groups, domain, post, biases):
    return {v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0 for g in q_groups for v in g}

def system_best_response_quadratic(q_groups, domain, post, biases, gamma=1.0):
    out = {}
    for g in q_groups:
        for v in g:
            a = gamma * post.get(v, 0.0) + biases.get(v, 0.0)
            out[v] = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return out

def system_best_response(q_groups, domain, prior, biases, receiver_model="threshold", gamma=1.0):
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold": return system_best_response_threshold(q_groups, domain, post, biases)
    if receiver_model == "quadratic": return system_best_response_quadratic(q_groups, domain, post, biases, gamma=gamma)
    raise ValueError("receiver_model must be 'threshold' or 'quadratic'")

def _exposure_weights(order_list, scheme="harmonic"):
    n = len(order_list)
    if n == 0: return {}
    if scheme == "harmonic": raw = [1.0 / (r + 1) for r in range(n)]
    elif scheme == "geometric": raw = [0.9 ** r for r in range(n)]
    else: raw = [1.0 / (r + 1) for r in range(n)]
    Z = sum(raw) or 1.0
    ws = [w / Z for w in raw]
    return {order_list[r]: ws[r] for r in range(n)}

def user_utility_from_response(theta, response, receiver_model, *, order_list=None, eps_order=0.0, exposure_scheme="harmonic"):
    if receiver_model == "threshold":
        base = sum(theta.get(v, 0.0) * response.get(v, 0.0) for v in theta.keys())
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

def flatten_in_order(q_groups): return [x for g in q_groups for x in g]

def op_pairs_strict(q_groups):
    pairs = []
    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def swap_single_pair(q_groups, u, v):
    return [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]

def dedupe_and_sort_desc(values): return sorted(set(values), reverse=True)

def _validate_partition_or_die(q_groups, domain):
    flat = [x for g in q_groups for x in g]
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_groups must be a partition of domain.")

# =============================================================================
# Alg. 1, 2, 4
# =============================================================================

def algorithm_1_credibility_detection(q_groups, domain, prior, *, biases, receiver_model="threshold", gamma=1.0, eps_order=0.0, rule="paper", tie_policy="neutral"):
    C = set()
    def _decide(a, b):
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
        up_q = user_utility_from_response(theta_plus, beta_q, receiver_model, order_list=order_q, eps_order=0.0)
        up_s = user_utility_from_response(theta_plus, beta_s, receiver_model, order_list=order_s, eps_order=0.0)
        um_q = user_utility_from_response(theta_min,  beta_q, receiver_model, order_list=order_q, eps_order=0.0)
        um_s = user_utility_from_response(theta_min,  beta_s, receiver_model, order_list=order_s, eps_order=0.0)
        if rule == "original":
            cplus  = _decide(up_q, up_s)
            cminus = _decide(um_q, um_s)
            if cplus is not None and cminus is not None and cplus != cminus:
                C.add((u, v))
        else:  # paper rule
            both_q    = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
            both_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
            if not (both_q or both_swap):
                C.add((u, v))
    return C

def build_adj(items, edges):
    adj = {x: [] for x in items}
    for u, v in edges:
        if u in adj and v in adj: adj[u].append(v)
    return adj

def reachable(start, adj):
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for w in adj.get(u, []):
            if w not in seen: seen.add(w); dq.append(w)
    return seen

def algorithm_2_build_qbase(initial_order, domain, prior, *, biases, receiver_model="threshold", gamma=1.0, eps_order=0.0, rule="paper"):
    q_cur = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=0.0, rule=rule)
        adj = build_adj(domain, C)
        merged, i = False, 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True; break
            i += 1
        if not merged: break
    return q_cur

def algorithm_2_build_qbase_from_groups(q_init_groups, domain, prior, *, biases, receiver_model="threshold", gamma=1.0, eps_order=0.0, rule="paper"):
    _validate_partition_or_die(q_init_groups, domain)
    q_cur = [g[:] for g in q_init_groups]
    for _ in range(max(0, len(q_cur) - 1) + 1):
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=0.0, rule=rule)
        adj = build_adj(domain, C)
        merged, i = False, 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not all(v in reachable(u, adj) for u in Gi for v in Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True; break
            i += 1
        if not merged: break
    return q_cur

def algorithm_4_maximally_informative(q_base, domain, prior, *, biases, receiver_model="threshold", gamma=1.0, eps_order_for_tiebreak=0.0):
    m = len(q_base)
    if m == 0: return []
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model, gamma=gamma)
    base_gain0, base_gain_eps = [], []
    for t in range(m):
        items_t = q_base[t]
        resp_t = {x: beta_base.get(x, 0 if receiver_model == "threshold" else 0.0) for x in items_t}
        u0 = user_utility_from_response(theta, resp_t, receiver_model, order_list=None, eps_order=0.0)
        ueps = user_utility_from_response(theta, resp_t, receiver_model, order_list=items_t, eps_order=eps_order_for_tiebreak)
        base_gain0.append(u0); base_gain_eps.append(ueps)
    C0 = [[0.0] * m for _ in range(m)]
    Ceps = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            run_items = [x for g in q_base[i:j+1] for x in g]
            temp_q = q_base[:i] + [run_items] + q_base[j+1:]
            beta_run = system_best_response(temp_q, domain, prior, biases, receiver_model, gamma=gamma)
            resp_run = {x: beta_run.get(x, 0 if receiver_model == "threshold" else 0.0) for x in run_items}
            u0_run   = user_utility_from_response(theta, resp_run, receiver_model, order_list=None, eps_order=0.0)
            ueps_run = user_utility_from_response(theta, resp_run, receiver_model, order_list=run_items, eps_order=eps_order_for_tiebreak)
            base_line0  = sum(base_gain0[t]   for t in range(i, j+1))
            base_lineep = sum(base_gain_eps[t] for t in range(i, j+1))
            C0[i][j]   = u0_run   - base_line0
            Ceps[i][j] = ueps_run - base_lineep
    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base
    dp, tie, prev = [0.0] * (m + 1), [0.0] * (m + 1), [-1] * (m + 1)
    for t in range(1, m + 1):
        best_val, best_tie, arg = -float("inf"), -float("inf"), -1
        for i in range(1, t + 1):
            g0, geps = C0[i-1][t-1], Ceps[i-1][t-1]
            val  = dp[i-1] + (g0   if g0   > EPS_COMPARE else 0.0)
            tiev = tie[i-1] + (geps if g0   > EPS_COMPARE else 0.0)
            better = (val > best_val + EPS_COMPARE) or \
                     (abs(val - best_val) <= EPS_COMPARE and (tiev > best_tie + EPS_COMPARE)) or \
                     (abs(val - best_val) <= EPS_COMPARE and abs(tiev - best_tie) <= EPS_COMPARE and (i - 1) > arg)
            if better: best_val, best_tie, arg = val, tiev, i - 1
        dp[t], tie[t], prev[t] = best_val, best_tie, arg
    q_star, t = [], m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(domain_values, prior, bias_obj, receiver_model="threshold", gamma=1.0, q_user=None, algorithm1_rule="paper"):
    domain = dedupe_and_sort_desc(domain_values)
    biases = biases_from_bias_obj(domain, bias_obj)
    if q_user is None:
        q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=EPS_ORDER, rule=algorithm1_rule)
    else:
        _validate_partition_or_die(q_user, domain)
        q_base = algorithm_2_build_qbase_from_groups(q_user, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order=EPS_ORDER, rule=algorithm1_rule)
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, receiver_model=receiver_model, gamma=gamma, eps_order_for_tiebreak=EPS_ORDER)
    return {"domain": domain, "q_user": q_user, "q_base": q_base, "q_star": q_star, "biases": biases}

def run_pipeline_domain(domain_values, prior, bias_obj, receiver_model="threshold", gamma=1.0, algorithm1_rule="paper"):
    return run_pipeline(domain_values, prior, bias_obj, receiver_model=receiver_model, gamma=gamma, q_user=None, algorithm1_rule=algorithm1_rule)

# =============================================================================
# Bucket utilities
# =============================================================================

def quantile_buckets(values: List[float], B: int = 4):
    if B <= 0: raise ValueError("B must be >= 1")
    arr = np.array(values, dtype=float)
    qs = np.linspace(0, 1, B + 1)
    cuts = np.quantile(arr, qs)
    cuts[0] -= 1e-9; cuts[-1] += 1e-9
    idx = np.searchsorted(cuts, arr, side='right') - 1
    idx = np.clip(idx, 0, B - 1)
    buckets = {}
    for v, b in zip(arr, idx):
        buckets.setdefault(int(b), []).append(float(v))
    rep = {b: float(np.mean(buckets[b])) for b in buckets}
    rep_to_items = {rep[b]: sorted(buckets[b], reverse=True) for b in buckets}
    bucket_reps = sorted(rep.values(), reverse=True)
    return bucket_reps, rep_to_items

class BiasFromMap:
    def __init__(self, mapping: Dict[float, float], default: float = 0.0):
        self.mapping = dict(mapping); self.default = float(default)
    def bias_for_value(self, x, _): return float(self.mapping.get(float(x), self.default))

def expand_bucket_query_to_items(q_bucket, rep_to_items):
    expanded = []
    for g in q_bucket:
        big = []
        for rep in g: big.extend(rep_to_items.get(rep, []))
        expanded.append(big)
    return expanded

def evaluate_query_item_level(q_groups_items, domain_items, prior, item_biases, receiver_model="threshold"):
    theta = compute_expected_posteriors(q_groups_items, domain_items, prior)
    beta  = system_best_response(q_groups_items, domain_items, prior, item_biases, receiver_model)
    return user_utility_from_response(theta, beta, receiver_model)

# =============================================================================
# Pretty formatter
# =============================================================================
def fmt_groups(q_groups: List[List[float]], nd=3):
    return " ≺ ".join("{" + ", ".join(map(lambda x: f"{x:.{nd}g}", g)) + "}" for g in q_groups)
# =============================================================================

def build_item_bias(values):
    arr = np.array(values, dtype=float)
    s = 1.0 / (1.0 + np.exp(-(arr - np.mean(arr)) / (0.18 * (np.max(arr)-np.min(arr)+1e-9))))
    base = 0.15 + 0.55 * s
    center = np.quantile(arr, 0.8)
    bump = np.exp(-((arr - center)**2) / (2 * (0.08 * (np.max(arr)-np.min(arr)+1e-9))**2))
    out = np.clip(base + 0.2 * bump, 0.0, 1.0)
    #make same bias all bias 0.49 + noise
    out=[0.80 for _ in arr]
    return {float(v): float(out[i]) for i, v in enumerate(arr)}

rng = np.random.default_rng(4242)
values = sorted(rng.uniform(1, 70, size=70).tolist(), reverse=True)
prior = PriorSpec(kind="uniform")
item_bias_true = build_item_bias(values)

class _BiasMap:
    def __init__(self, m): self.m = m
    def bias_for_value(self, x, _): return self.m[float(x)]

# Compute item-level credibility q_base
out_item = run_pipeline(values, prior, _BiasMap(item_bias_true), receiver_model="threshold", algorithm1_rule="paper")
q_base_items = out_item["q_base"]
domain_items = dedupe_and_sort_desc(values)

# q* (full) DP on q_base
q_star_full = algorithm_4_maximally_informative(q_base_items, domain_items, prior, biases=item_bias_true, receiver_model="threshold")

# q**_samecred: bucket-averaged biases on the SAME q_base (B=6)
bucket_reps, rep_to_items = quantile_buckets(domain_items, B=6)
bucket_avg = {rep: float(np.mean([item_bias_true[v] for v in rep_to_items[rep]])) for rep in bucket_reps}
item2rep = {v: rep for rep, items in rep_to_items.items() for v in items}
item_bias_bucket = {v: bucket_avg[item2rep[v]] for v in domain_items}
q_star_bucket_samecred = algorithm_4_maximally_informative(q_base_items, domain_items, prior, biases=item_bias_bucket, receiver_model="threshold")

# Evaluate both on item-level
u_full  = evaluate_query_item_level(q_star_full,  domain_items, prior, item_bias_true, receiver_model="threshold")
u_bsame = evaluate_query_item_level(q_star_bucket_samecred, domain_items, prior, item_bias_true, receiver_model="threshold")

print("=== Same-Credibility Demo (B=6) ===")
print("q_base (items):", fmt_groups(q_base_items, nd=3))
print("\nq* (full-info):", fmt_groups(q_star_full, nd=3))
print("q** (bucket avg on same q_base):", fmt_groups(q_star_bucket_samecred, nd=3))
print("\nItem-level utility:")
print(f"  U(q*)      = {u_full:.6f}")
print(f"  U(q**_SC)  = {u_bsame:.6f}")
print(f"  Δ = U(q*) - U(q**_SC) = {u_full - u_bsame:.6f}")