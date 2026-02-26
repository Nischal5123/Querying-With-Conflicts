# Scalability experiment scaffold for Alg.1 (credibility) and Alg.2+4 (maximally-informative)
# across 1/2/3 ranking attributes using the census dataset (or a synthetic fallback).
#
# - No bucketing; bias is defined on each Cartesian-product domain value (tuple).
# - We measure time for Alg.1 and Alg.2+4; in the plot, Alg.2+4 is shown as a single curve (sum of times).
# - We save a CSV with timings and metadata, plus a PNG plot.
# - Includes a 15-minute per-task timeout using multiprocessing.
# - For this demo run (to finish quickly here), we cap unique levels per attribute; for your full run, set caps to None.
#
import os, math, time, json, multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Iterable
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


# -------------------------------
# Core algorithms (uniform prior)
# -------------------------------

EPS_COMPARE = 1e-5
EPS_ORDER   = 1e-9

@dataclass
class PriorSpec:
    kind: str = "uniform"
    def rank_expectation(self, k: int, r_desc: int) -> float:
        # Closed form uniform
        j = k - r_desc + 1
        return j / (k + 1.0)




def compute_expected_posteriors(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec) -> Dict[Any, float]:
    k = len(domain)
    post: Dict[Any, float] = {}
    r = 1
    for g in q_groups:
        n = len(g)
        if n <= 0:
            continue
        # Block average (uniform)
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a = ((j_hi + j_lo) / 2.0) / (k + 1.0)
        for v in g:
            post[v] = float(a)
        r += n
    for v in domain:
        post.setdefault(v, 0.0)
    return post

def system_best_response_threshold(q_groups, domain, post, biases) -> Dict[Any, int]:
    return {v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0
            for g in q_groups for v in g}

def system_best_response(q_groups, domain, prior, biases, receiver_model="threshold", **kw):
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    raise ValueError("Only 'threshold' implemented in this experiment runner.")

def user_utility_from_response(theta: Dict[Any,float], response: Dict[Any,float]) -> float:
    return sum(theta.get(v, 0.0) * response.get(v, 0.0) for v in theta.keys())

def flatten_in_order(q_groups: List[List[Any]]) -> List[Any]:
    return [x for g in q_groups for x in g]

def op_pairs_strict(q_groups: List[List[Any]], pair_span: Optional[int]=None) -> List[Tuple[Any, Any]]:
    # If pair_span is provided, only include pairs within that forward span (to avoid collapse / reduce cost).
    pairs: List[Tuple[Any, Any]] = []
    for i in range(len(q_groups)):
        jmax = len(q_groups) if pair_span is None else min(len(q_groups), i + 1 + pair_span)
        for j in range(i + 1, jmax):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    pairs.append((u, v))
    return pairs

def swap_single_pair(q_groups: List[List[Any]], u: Any, v: Any) -> List[List[Any]]:
    return [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]

def dedupe_and_sort_desc(values: Iterable[Any]) -> List[Any]:
    # Python's default tuple ordering is lexicographic; we reverse for DESC
    return sorted(set(values), reverse=True)

# def algorithm_1_credibility_detection(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
#                                       biases: Dict[Any,float], receiver_model: str = "threshold",
#                                       rule: str = "paper", pair_span: Optional[int]=None) -> List[Tuple[Any, Any]]:
#     """
#     Paper rule: Mark (u,v) credible unless both types strictly prefer the same column.
#     Here we use threshold receiver and no epsilon bonus.
#     """
#     C: List[Tuple[Any, Any]] = []
#
#     theta_plus = compute_expected_posteriors(q_groups, domain, prior)
#     beta_q     = system_best_response(q_groups, domain, prior, biases, receiver_model)
#     # For each strict pair, check 2x2 utilities
#     def decide(up_q, up_s, um_q, um_s) -> bool:
#         both_q    = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
#         both_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
#         return not (both_q or both_swap)
#
#     for (u,v) in op_pairs_strict(q_groups, pair_span=pair_span):
#         q_swap    = swap_single_pair(q_groups, u, v)
#         theta_min = compute_expected_posteriors(q_swap, domain, prior)
#         beta_s    = system_best_response(q_swap, domain, prior, biases, receiver_model)
#
#         up_q = user_utility_from_response(theta_plus, beta_q)
#         up_s = user_utility_from_response(theta_plus, beta_s)
#         um_q = user_utility_from_response(theta_min,  beta_q)
#         um_s = user_utility_from_response(theta_min,  beta_s)
#
#         if decide(up_q, up_s, um_q, um_s):
#             C.append((u, v))
#     return C

def _flatten(q_groups: List[List[Any]]) -> List[Any]:
    return [x for g in q_groups for x in g]

def _group_posteriors_uniform(q_groups: List[List[Any]], k: int) -> List[float]:
    """
    For uniform prior, posterior for a group occupying ranks r..r+n-1 in DESC order is:
        a_g = average_{rr=r..r+n-1} ( (k - rr + 1) / (k+1) )
            = ((j_hi + j_lo)/2) / (k+1)
       where j_hi = k - r + 1, j_lo = k - (r+n-1) + 1
    Returns a list a_g per group g in q_groups.
    """
    a = []
    r = 1
    for g in q_groups:
        n = len(g)
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a.append(((j_hi + j_lo) / 2.0) / (k + 1.0))
        r += n
    return a

def algorithm_1_credibility_detection(q_groups, domain, prior, *, biases, pair_span=None) -> List[Tuple[Any, Any]]:
    # Map items to indices and groups
    idx = {v:i for i,v in enumerate([x for g in q_groups for x in g])}
    pos = [None]*len(idx)               # index -> (group_index, local_index)
    items = [x for g in q_groups for x in g]
    gid_of = [None]*len(items)          # index -> group index
    start = 0
    group_as = []                       # a_g per group
    k = len(domain)
    r = 1
    for g_i, g in enumerate(q_groups):
        n = len(g)
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a = ((j_hi + j_lo)/2.0)/(k+1.0)
        group_as.append(a)
        for j, v in enumerate(g):
            i = idx[v]
            gid_of[i] = g_i
            pos[i] = (g_i, j)
        r += n

    # Precompute theta+ and beta_q and baseline U++
    theta_plus = [group_as[gid_of[i]] for i in range(len(items))]
    beta_q = [1 if (theta_plus[i] - biases[items[i]]) > EPS_COMPARE else 0
              for i in range(len(items))]
    Upp = sum(theta_plus[i]*beta_q[i] for i in range(len(items)))

    C = []
    # iterate inter-group pairs
    for gi in range(len(q_groups)):
        for gj in range(gi+1, len(q_groups)):
            for u in q_groups[gi]:
                iu = idx[u]
                for v in q_groups[gj]:
                    iv = idx[v]
                    ai, aj = group_as[gi], group_as[gj]
                    bu, bv = biases[u], biases[v]

                    # post-swap actions at u, v
                    betas_u = 1 if (aj - bu) > EPS_COMPARE else 0
                    betas_v = 1 if (ai - bv) > EPS_COMPARE else 0

                    # up_s: only actions at u,v may change
                    up_s = Upp \
                         + theta_plus[iu]* (betas_u - beta_q[iu]) \
                         + theta_plus[iv]* (betas_v - beta_q[iv])
                    up_q = Upp

                    # theta- swaps thetas of u and v
                    um_q = Upp \
                         + (theta_plus[iv]-theta_plus[iu]) * beta_q[iu] \
                         + (theta_plus[iu]-theta_plus[iv]) * beta_q[iv]
                    um_s = up_s \
                         + (theta_plus[iv]-theta_plus[iu]) * betas_u \
                         + (theta_plus[iu]-theta_plus[iv]) * betas_v

                    # Paper rule: credible unless both rows strictly prefer same col
                    both_q    = (up_q > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
                    both_swap = (up_s > up_q + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
                    if not (both_q or both_swap):
                        C.append((u, v))
    return C

def _credibility_edges_paper_fast(
    q_groups: List[List[Any]],
    domain: List[Any],
    biases: Dict[Any, float],
) -> List[Tuple[Any, Any]]:
    """
    Return credible strict edges (u,v) present in q_groups (inter-group only),
    using the paper rule:
        credible iff NOT( both types strictly prefer q OR both strictly prefer swap ).
    Threshold receiver; epsilon=0; uniform prior.
    """
    items = _flatten(q_groups)
    k = len(domain)

    # Map item -> index and index -> group id
    idx = {v: i for i, v in enumerate(items)}
    gid_of = [None] * len(items)
    for gi, g in enumerate(q_groups):
        for v in g:
            gid_of[idx[v]] = gi

    # Group posteriors and per-item θ⁺ and β(q)
    a_g = _group_posteriors_uniform(q_groups, k)
    theta_plus = [a_g[gid_of[i]] for i in range(len(items))]
    beta_q = [1 if (theta_plus[i] - biases[items[i]]) > EPS_COMPARE else 0
              for i in range(len(items))]
    # Baseline U(θ⁺, β(q))
    U_pp = sum(theta_plus[i] * beta_q[i] for i in range(len(items)))

    C = []
    # Iterate strict inter-group pairs (gi < gj)
    for gi in range(len(q_groups)):
        for gj in range(gi + 1, len(q_groups)):
            ai, aj = a_g[gi], a_g[gj]
            for u in q_groups[gi]:
                iu = idx[u]
                bu = biases[u]
                # β under swap at u (moves to group gj)
                betas_u = 1 if (aj - bu) > EPS_COMPARE else 0
                for v in q_groups[gj]:
                    iv = idx[v]
                    bv = biases[v]
                    # β under swap at v (moves to group gi)
                    betas_v = 1 if (ai - bv) > EPS_COMPARE else 0

                    # up_q = U_pp
                    # up_s: only actions at u, v may change
                    up_s = U_pp \
                         + theta_plus[iu] * (betas_u - beta_q[iu]) \
                         + theta_plus[iv] * (betas_v - beta_q[iv])

                    # θ⁻ (for the swapped query) differs only at u and v:
                    # θ⁻(u) = aj, θ⁻(v) = ai
                    # Use deltas on U_pp:
                    um_q = U_pp \
                         + (aj - theta_plus[iu]) * beta_q[iu] \
                         + (ai - theta_plus[iv]) * beta_q[iv]
                    um_s = up_s \
                         + (aj - theta_plus[iu]) * betas_u \
                         + (ai - theta_plus[iv]) * betas_v

                    both_q    = (U_pp > up_s + EPS_COMPARE) and (um_q > um_s + EPS_COMPARE)
                    both_swap = (up_s > U_pp + EPS_COMPARE) and (um_s > um_q + EPS_COMPARE)
                    if not (both_q or both_swap):
                        C.append((u, v))
    return C

# -----------------------------------------------

# def build_adj(items: List[Any], edges: List[Tuple[Any, Any]]) -> Dict[Any, List[Any]]:
#     adj = {x: [] for x in items}
#     for u, v in edges:
#         if u in adj and v in adj:
#             adj[u].append(v)
#     return adj

# def reachable(start: Any, adj: Dict[Any, List[Any]]) -> set:
#     seen, dq = {start}, deque([start])
#     while dq:
#         u = dq.popleft()
#         for w in adj.get(u, []):
#             if w not in seen:
#                 seen.add(w)
#                 dq.append(w)
#     return seen

# def algorithm_2_build_qbase(initial_order: List[Any], domain: List[Any], prior: PriorSpec, *,
#                             biases: Dict[Any,float], receiver_model: str = "threshold",
#                             rule: str="paper", pair_span: Optional[int]=None) -> List[List[Any]]:
#     q_cur: List[List[Any]] = [[x] for x in initial_order]
#     for _ in range(len(domain) + 1):
#         C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases,
#                                               receiver_model=receiver_model, rule=rule, pair_span=pair_span)
#         adj = build_adj(domain, C)
#         merged = False
#         i = 0
#         while i < len(q_cur) - 1:
#             Gi, Gj = q_cur[i], q_cur[i + 1]
#             # Boundary Gi->Gj must be supported for ALL u∈Gi, v∈Gj
#             if not all(v in reachable(u, adj) for u in Gi for v in Gj):
#                 q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
#                 merged = True
#                 break
#             i += 1
#         if not merged:
#             break
#     return q_cur

# -----------------------------------------------

def _reachability_bitsets(items: List[Any], edges: List[Tuple[Any, Any]]) -> Tuple[Dict[Any, int], List[int]]:
    """
    items: in presentation (query) order
    edges: credible strict edges (u,v) with u before v
    Returns:
      idx: value -> position index
      R:   list of bitsets; R[i] has bits for all j reachable from i
    """
    n = len(items)
    idx = {v: i for i, v in enumerate(items)}
    neigh = [[] for _ in range(n)]
    for u, v in edges:
        iu, iv = idx[u], idx[v]
        if iu < iv:  # keep forward edges
            neigh[iu].append(iv)

    R = [0] * n
    # reverse order DP: R[i] = union over neighbors j of (bit j) | R[j]
    for i in range(n - 1, -1, -1):
        mask = 0
        for j in neigh[i]:
            mask |= (1 << j) | R[j]
        R[i] = mask
    return idx, R

def _boundary_supported(idx_map: Dict[Any, int], R: List[int], Gi: List[Any], Gj: List[Any]) -> bool:
    want = 0
    for v in Gj:
        want |= (1 << idx_map[v])
    for u in Gi:
        if (R[idx_map[u]] & want) != want:
            return False
    return True

# --- Algorithm 2 (start from singletons) ---

def algorithm_2_build_qbase(
    initial_order: List[Any],
    domain: List[Any],
    prior: PriorSpec,  # kept for API symmetry; uniform closed-form is used
    *,
    biases: Dict[Any, float], pair_span: Optional[int]=None,
) -> List[List[Any]]:
    """
    Alg. 2: Start from SINGLETONS (strict order) and merge adjacent boundaries
    that are NOT supported by credibility reachability (paper rule).
    - Threshold receiver, uniform prior.
    - No heuristics; identical outputs to the textbook Alg.2.

    Returns q_base as a list of groups in DESC presentation order.
    """
    # 1) Start from strict singletons
    q_cur: List[List[Any]] = [[x] for x in initial_order]

    # 2) Iterate until stable
    for _ in range(len(domain) + 1):
        # 2a) Compute credible edges on the CURRENT grouping (paper rule)
        C = _credibility_edges_paper_fast(q_cur, domain, biases)
        # 2b) Build reachability bitsets
        idx_map, R = _reachability_bitsets(_flatten(q_cur), C)

        # 2c) Scan boundaries left-to-right; merge the first unsupported boundary
        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported(idx_map, R, Gi, Gj):
                # merge Gi and Gj
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1

        if not merged:
            break

    return q_cur

# --- Variant: Algorithm 2 starting from user-provided groups (ties respected) ---

def algorithm_2_build_qbase_from_groups(
    q_init_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any, float],
) -> List[List[Any]]:
    """
    Alg. 2 (group-respecting): Start from a USER-PROVIDED PARTIAL RANKING (ties intact),
    and merge adjacent group boundaries that are unsupported by credibility reachability.
    - Never splits a user tie.
    - Threshold receiver, uniform prior, paper rule.
    """
    # quick partition check
    flat = _flatten(q_init_groups)
    if set(flat) != set(domain) or len(flat) != len(domain):
        raise ValueError("q_init_groups must be a partition of domain (cover each value exactly once).")

    q_cur = [g[:] for g in q_init_groups]

    for _ in range(max(0, len(q_cur) - 1) + 1):
        C = _credibility_edges_paper_fast(q_cur, domain, biases)
        idx_map, R = _reachability_bitsets(_flatten(q_cur), C)

        merged = False
        i = 0
        while i < len(q_cur) - 1:
            Gi, Gj = q_cur[i], q_cur[i + 1]
            if not _boundary_supported(idx_map, R, Gi, Gj):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:]
                merged = True
                break
            i += 1
        if not merged:
            break

    return q_cur

# --- Fast credibility edge computation (paper rule, threshold receiver) ---

def algorithm_4_maximally_informative(q_base: List[List[Any]], domain: List[Any], prior: PriorSpec, *,
                                      biases: Dict[Any,float], receiver_model: str="threshold") -> List[List[Any]]:
    m = len(q_base)
    if m == 0:
        return []
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta_base = system_best_response(q_base, domain, prior, biases, receiver_model)
    base_gain0: List[float] = []
    for t in range(m):
        items_t = q_base[t]
        resp_t = {x: beta_base.get(x, 0) for x in items_t}
        u0 = sum(theta[v] * resp_t.get(v, 0) for v in items_t)
        base_gain0.append(u0)

    C0 = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            run_items = [x for g in q_base[i:j+1] for x in g]
            temp_q = q_base[:i] + [run_items] + q_base[j+1:]
            beta_run = system_best_response(temp_q, domain, prior, biases, receiver_model)
            u0_run = sum(theta[v] * beta_run.get(v, 0) for v in run_items)
            base_line0 = sum(base_gain0[t] for t in range(i, j+1))
            C0[i][j] = u0_run - base_line0

    if max(C0[i][j] for i in range(m) for j in range(i, m)) <= EPS_COMPARE:
        return q_base

    dp = [0.0]*(m+1)
    tie = [0.0]*(m+1)
    prev = [-1]*(m+1)

    for t in range(1, m+1):
        best_val, arg = -1e18, -1
        for i in range(1, t+1):
            g0 = C0[i-1][t-1]
            val  = dp[i-1] + (g0 if g0 > EPS_COMPARE else 0.0)
            better = (val > best_val + EPS_COMPARE) or (abs(val - best_val) <= EPS_COMPARE and (i - 1) > arg)
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

# -------------------------------
# Bias construction (monotone-ish; ≳95% above 0.5)
# -------------------------------

def build_bias_on_cartesian(domain: List[Tuple], attr_names: List[str], mins: Dict[str,float], maxs: Dict[str,float]) -> Dict[Tuple, float]:
    """
    Monotone asymmetric: logistic of a weighted sum of normalized attributes.
    Calibrate shift so ≈95% of domain have bias > 0.5.
    """
    # Weights: emphasize first attribute, then others
    w = []
    for i, a in enumerate(attr_names):
        if a == "sex":
            w.append(0.4)   # small influence
        elif i == 0:
            w.append(1.0)   # main
        else:
            w.append(0.6)   # secondary
    w = np.array(w, dtype=float)

    # Compute linear scores
    scores = []
    for tup in domain:
        xs = []
        for (a, x) in zip(attr_names, tup):
            if a == "sex":
                # assume sex encoded 0/1 already
                xs.append(float(x))
            else:
                lo, hi = mins[a], maxs[a]
                t = 0.0 if hi == lo else (float(x) - lo)/(hi - lo)
                xs.append(t)
        scores.append(float(np.dot(w, np.array(xs))))
    scores = np.array(scores)
    # Calibrate shift so 95th percentile threshold is at 0.5 in logistic space
    # logistic(k*(score - c)) ; choose k so transition is reasonably sharp, and c as 5th percentile
    kappa = 8.0
    c = np.quantile(scores, 0.05)  # so ~95% have score > c => bias > 0.5
    logits = kappa*(scores - c)
    bias = 1.0/(1.0 + np.exp(-logits))
    return {dom: float(min(max(b, 0.0), 1.0)) for dom, b in zip(domain, bias)}

# -------------------------------
# Data prep
# -------------------------------

def load_census_df():
    candidate = "../data/real/census.csv"
    if os.path.exists(candidate):
        try:
            df = pd.read_csv(candidate)
            return df, True
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

def distinct_values(df: pd.DataFrame, col: str, max_levels: Optional[int]=None) -> List:
    xs = sorted(set(df[col].tolist()), reverse=True)
    if max_levels is not None and len(xs) > max_levels:
        xs = xs[:max_levels]
    return xs

def build_cartesian_domain(df: pd.DataFrame, attrs: List[str], caps: Dict[str, Optional[int]]) -> Tuple[List[Tuple], Dict[str,float], Dict[str,float]]:
    values_lists = []
    mins, maxs = {}, {}
    for a in attrs:
        vals = distinct_values(df, a, max_levels=caps.get(a))
        values_lists.append(vals)
        mins[a] = float(min(vals))
        maxs[a] = float(max(vals))
    # Cartesian product (DESC lexicographic because each list is DESC)
    domain = list(product(*values_lists))
    domain = sorted(domain, reverse=True)
    return domain, mins, maxs

# -------------------------------
# Timing with timeout
# -------------------------------

def _time_alg1(domain, prior, biases, pair_span=None, return_dict=None):
    print("Timing Alg.1 on domain size", len(domain))
    start = time.perf_counter()
    q0 = [[x] for x in domain]  # strict order
    C = algorithm_1_credibility_detection(q0, domain, prior, biases=biases, pair_span=pair_span)
    dur = time.perf_counter() - start
    if return_dict is not None:
        return_dict.update({"cred_edges": len(C), "time": dur})

def _time_alg24(domain, prior, biases, pair_span=None, return_dict=None):
    print("Timing Alg.2+4 on domain size", len(domain))
    t0 = time.perf_counter()
    q_base = algorithm_2_build_qbase(domain, domain, prior, biases=biases, pair_span=pair_span)
    t2 = time.perf_counter()
    q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases)
    t4 = time.perf_counter()
    if return_dict is not None:
        return_dict.update({
            "q_base_groups": len(q_base),
            "q_star_groups": len(q_star),
            "time2": t2 - t0,
            "time4": t4 - t2,
            "time24": t4 - t0,
        })

def run_with_timeout(func, args=(), kwargs=None, timeout_s=900):
    if kwargs is None: kwargs = {}
    mgr = mp.Manager()
    ret = mgr.dict()
    kwargs = dict(kwargs)
    kwargs["return_dict"] = ret
    p = mp.Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, True  # None result, timed out
    return dict(ret), False



# -------------------------------
# Experiment runner
# -------------------------------

def run_scalability_experiment(caps_for_demo: Dict[str, Optional[int]] = None, pair_span: Optional[int]=None):
    """
    caps_for_demo: optional cap on unique levels for each attribute for a quick demo.
    pair_span: limit Alg.1 pair comparisons to nearby groups to keep demo quick; set None for full.
    """
    df_raw, found = load_census_df()
    df = encode_attributes(df_raw)
    # Attribute sets
    attr_sets = [
        ["sex"],
        ["education_num", "sex"],
        ["education_num", "sex", "age"]

    ]
    caps = caps_for_demo or {"age": None, "education_num": None, "sex": None}

    prior = PriorSpec()
    records = []
    for attrs in attr_sets:
        domain, mins, maxs = build_cartesian_domain(df, attrs, caps)
        biases = build_bias_on_cartesian(domain, attrs, mins, maxs)
        # Time Alg.1
        r1, to1 = run_with_timeout(_time_alg1, args=(domain, prior, biases), kwargs={"pair_span": pair_span}, timeout_s=900)  # 60s for demo
        # Time Alg.2+4
        r24, to24 = run_with_timeout(_time_alg24, args=(domain, prior, biases), kwargs={"pair_span": pair_span}, timeout_s=900)  # 60s for demo

        rec = {
            "attrs": "+".join(attrs),
            "num_attrs": len(attrs),
            "domain_size": len(domain),
            "alg1_time_s": (None if (r1 is None) else r1.get("time", None)),
            "alg1_timeout": to1,
            "alg1_cred_edges": (None if (r1 is None) else r1.get("cred_edges", None)),
            "alg2_time_s": (None if (r24 is None) else r24.get("time2", None)),
            "alg4_time_s": (None if (r24 is None) else r24.get("time4", None)),
            "alg24_time_s": (None if (r24 is None) else r24.get("time24", None)),
            "alg24_timeout": to24,
            "q_base_groups": (None if (r24 is None) else r24.get("q_base_groups", None)),
            "q_star_groups": (None if (r24 is None) else r24.get("q_star_groups", None)),
            "data_source": "real_census" if found else "synthetic_fallback",
        }
        records.append(rec)

    res = pd.DataFrame.from_records(records).sort_values("num_attrs")
    out_csv = "../results/exp_scalability_results.csv"
    res.to_csv(out_csv, index=False)

    # Plot
    fig = plt.figure(figsize=(6,4))
    xs = res["num_attrs"].tolist()
    y1 = [res.loc[i, "alg1_time_s"] if pd.notnull(res.loc[i, "alg1_time_s"]) else float("nan") for i in res.index]
    y24 = [res.loc[i, "alg24_time_s"] if pd.notnull(res.loc[i, "alg24_time_s"]) else float("nan") for i in res.index]
    plt.plot(xs, y1, marker="o", label="Alg.1 Credibility")
    plt.plot(xs, y24, marker="o", label="Alg.2+4 Maximally Informative")
    plt.xlabel("# Attributes")
    plt.ylabel("Time (s)")
    plt.title("Scalability vs # Attributes (demo)")
    plt.legend()
    plt.tight_layout()
    out_png = "../results/exp_scalability_plot.png"
    plt.savefig(out_png)
    plt.close(fig)


    print("Saved results CSV:", out_csv)
    print("Saved plot PNG:", out_png)

# -------------------------------
# Run a quick demo with caps to keep it fast here.
# For your full experiment, set caps_for_demo to None and timeout_s to 900 (15 min).
# -------------------------------
if __name__ == "__main__":
    run_scalability_experiment(
        caps_for_demo=None,  # cap unique levels just for this quick run
        pair_span=None  # limit Alg.1 to nearby pairs to speed up the demo; set None for full
    )
