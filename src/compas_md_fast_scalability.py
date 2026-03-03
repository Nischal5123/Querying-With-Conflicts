# -*- coding: utf-8 -*-
"""
COI (distinct-domain) + Scalability experiment over #attributes.

What you get:
- Core algorithms:
    * Alg.1  : Credibility detection (fast path for uniform+threshold, fallback for quadratic)
    * Alg.2  : Build q_base by merging unsupported boundaries
    * Alg.4  : Maximally-informative DP over q_base
- Utilities:
    * evaluate_plan_utility(...) for q_babble/q_base/q_star (supports quadratic)
- Tuple-safe bias:
    * make_random_multilevel_bias(...) works for tuple domains (Cartesian product)
- Experiment:
    * run_scalability_time_vs_num_attrs_logging(...)
      - Measures time and utility for THRESHOLD or QUADRATIC models.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Callable, Optional, Any
import itertools, json, os, time, math
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Global tolerances
# ---------------------------------------------------------------------
EPS_COMPARE = 1e-5  # numeric comparison tolerance
EPS_ORDER = 1e-9  # tiny order-aware bonus weight (presentational/tie-break)


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


def biases_from_bias_obj(domain: List[Any], bias_obj: Any) -> Dict[Any, float]:
    if not domain: return {}
    info = {"min": min(domain), "max": max(domain)}
    return {v: float(min(max(bias_obj.bias_for_value(v, info), 0.0), 1.0)) for v in domain}


def make_random_L_level_bias(domain_values, L, seed=123):
    rng = np.random.default_rng(seed)
    dom = list(dict.fromkeys(domain_values))
    levels = tuple(np.linspace(0.05, 0.95, num=L))
    labels = rng.choice(len(levels), size=len(dom))
    value2bias = {dom[i]: float(levels[int(lbl)]) for i, lbl in enumerate(labels)}
    return Bias1D(kind="custom", custom=lambda v, _: value2bias.get(v, 0.0))


# =============================================================================
# Receiver models & utilities
# =============================================================================

def system_best_response_threshold(q_groups, domain, post, biases):
    return {v: 1 if (post.get(v, 0.0) - biases.get(v, 0.0)) > EPS_COMPARE else 0 for v in domain}


def system_best_response_quadratic(q_groups, domain, post, biases, gamma=1.0):
    out = {}
    for v in domain:
        a = gamma * post.get(v, 0.0) + biases.get(v, 0.0)
        out[v] = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return out


def system_best_response(q_groups, domain, prior, biases, receiver_model="threshold", gamma=1.0):
    post = compute_expected_posteriors(q_groups, domain, prior)
    if receiver_model == "threshold":
        return system_best_response_threshold(q_groups, domain, post, biases)
    return system_best_response_quadratic(q_groups, domain, post, biases, gamma=gamma)


def user_utility_from_response(theta, response, receiver_model, *, order_list=None, eps_order=0.0):
    if receiver_model == "threshold":
        base = sum(theta.get(v, 0.0) * a for v, a in response.items())
    else:
        # Quadratic utility: - \sum (action - truth)^2
        base = -sum((response.get(v, 0.0) - theta.get(v, 0.0)) ** 2 for v in theta.keys())

    if eps_order > 0.0 and order_list is not None:
        # Simple exposure weight tie-breaker
        for r, v in enumerate(order_list):
            base += eps_order * (1.0 / (r + 1)) * theta.get(v, 0.0) * float(response.get(v, 0.0))
    return base


# =============================================================================
# Alg. 1, 2, 4
# =============================================================================

def algorithm_1_credibility_detection(
        q_groups, domain, prior, *, biases, receiver_model="threshold", gamma=1.0, eps_order=0.0, rule="paper"
) -> Set[Tuple[Any, Any]]:
    # Fast path for Uniform + Threshold
    if prior.kind == "uniform" and receiver_model == "threshold" and abs(eps_order) <= 1e-12:
        items = [x for g in q_groups for x in g];
        k = len(domain)
        idx = {v: i for i, v in enumerate(items)}
        gid_of = [0] * len(items)
        for gi, g in enumerate(q_groups):
            for v in g: gid_of[idx[v]] = gi
        a_g = []
        r = 1
        for g in q_groups:
            n = len(g);
            j_hi, j_lo = k - r + 1, k - (r + n - 1) + 1
            a_g.append(((j_hi + j_lo) / 2.0) / (k + 1.0));
            r += n
        theta_plus_vals = [a_g[gid_of[i]] for i in range(len(items))]
        beta_q_vals = [1 if (theta_plus_vals[i] - biases[items[i]]) > EPS_COMPARE else 0 for i in range(len(items))]
        U_pp = sum(theta_plus_vals[i] * beta_q_vals[i] for i in range(len(items)))

        C = set()
        for gi in range(len(q_groups)):
            for gj in range(gi + 1, len(q_groups)):
                ai, aj = a_g[gi], a_g[gj]
                for u in q_groups[gi]:
                    iu = idx[u];
                    bu = biases[u]
                    betas_u = 1 if (aj - bu) > EPS_COMPARE else 0
                    for v in q_groups[gj]:
                        iv = idx[v];
                        bv = biases[v]
                        betas_v = 1 if (ai - bv) > EPS_COMPARE else 0
                        up_s = U_pp + theta_plus_vals[iu] * (betas_u - beta_q_vals[iu]) + theta_plus_vals[iv] * (
                                    betas_v - beta_q_vals[iv])
                        um_q = U_pp + (aj - theta_plus_vals[iu]) * beta_q_vals[iu] + (ai - theta_plus_vals[iv]) * \
                               beta_q_vals[iv]
                        um_s = up_s + (aj - theta_plus_vals[iu]) * betas_u + (ai - theta_plus_vals[iv]) * betas_v
                        if not ((U_pp > up_s + EPS_COMPARE and um_q > um_s + EPS_COMPARE) or (
                                up_s > U_pp + EPS_COMPARE and um_s > um_q + EPS_COMPARE)):
                            C.add((u, v))
        return C

    # Generic Fallback (Supports Quadratic)
    C = set()
    theta_plus = compute_expected_posteriors(q_groups, domain, prior)
    beta_q = system_best_response(q_groups, domain, prior, biases, receiver_model, gamma)

    for i in range(len(q_groups)):
        for j in range(i + 1, len(q_groups)):
            for u in q_groups[i]:
                for v in q_groups[j]:
                    q_swap = [[(v if x == u else (u if x == v else x)) for x in g] for g in q_groups]
                    theta_min = compute_expected_posteriors(q_swap, domain, prior)
                    beta_s = system_best_response(q_swap, domain, prior, biases, receiver_model, gamma)
                    up_q = user_utility_from_response(theta_plus, beta_q, receiver_model)
                    up_s = user_utility_from_response(theta_plus, beta_s, receiver_model)
                    um_q = user_utility_from_response(theta_min, beta_q, receiver_model)
                    um_s = user_utility_from_response(theta_min, beta_s, receiver_model)
                    if not ((up_q > up_s + EPS_COMPARE and um_q > um_s + EPS_COMPARE) or (
                            up_s > up_q + EPS_COMPARE and um_s > um_q + EPS_COMPARE)):
                        C.add((u, v))
    return C


def _reachability_bitsets(items, edges):
    n = len(items);
    idx = {v: i for i, v in enumerate(items)}
    adj = [[] for _ in range(n)]
    for u, v in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is not None and iv is not None and iu < iv: adj[iu].append(iv)
    R = [0] * n
    for i in range(n - 1, -1, -1):
        m = 0
        for j in adj[i]: m |= (1 << j) | R[j]
        R[i] = m
    return idx, R


def algorithm_2_build_qbase(initial_order, domain, prior, *, biases, receiver_model="threshold", gamma=1.0,
                            rule="paper"):
    q_cur = [[x] for x in initial_order]
    for _ in range(len(domain) + 1):
        C = algorithm_1_credibility_detection(q_cur, domain, prior, biases=biases, receiver_model=receiver_model,
                                              gamma=gamma, rule=rule)
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)
        merged = False
        for i in range(len(q_cur) - 1):
            Gi, Gj = q_cur[i], q_cur[i + 1]
            want_mask = sum(1 << idx_map[v] for v in Gj)
            if not all((R[idx_map[u]] & want_mask) == want_mask for u in Gi):
                q_cur = q_cur[:i] + [Gi + Gj] + q_cur[i + 2:];
                merged = True;
                break
        if not merged: break
    return q_cur


def algorithm_4_maximally_informative(q_base, domain, prior, *, biases, receiver_model="threshold", gamma=1.0):
    m = len(q_base)
    if m == 0: return []
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    dp = [0.0] * (m + 1);
    prev = [-1] * (m + 1)
    for t in range(1, m + 1):
        best_val, arg = -float("inf"), -1
        for i in range(1, t + 1):
            run = [x for g in q_base[i - 1:t] for x in g]
            resp = system_best_response([run], domain, prior, biases, receiver_model, gamma)
            # Local utility for the items in the run
            u = user_utility_from_response({x: theta[x] for x in run}, resp, receiver_model)
            val = dp[i - 1] + u
            if val > best_val + EPS_COMPARE: best_val, arg = val, i - 1
        dp[t], prev[t] = best_val, arg
    q_star = [];
    t = m
    while t > 0:
        i = prev[t];
        q_star.insert(0, [x for g in q_base[i:t] for x in g]);
        t = i
    return q_star


# =============================================================================
# Scalability experiment
# =============================================================================

def build_cartesian_domain(df, cols):
    domains = [sorted(pd.unique(df[c]).tolist()) for c in cols]
    dom = [tuple(t) for t in itertools.product(*domains)]
    dom.sort(reverse=True)
    return dom


def pick_columns_smallest_domains(df, candidates, k):
    stats = sorted([(c, int(df[c].nunique())) for c in candidates], key=lambda t: (t[1], t[0]))
    return [c for c, _ in stats[:k]]


def evaluate_plan_utility(q_groups, domain, prior, *, biases, receiver_model="threshold", gamma=1.0):
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta = system_best_response(q_groups, domain, prior, biases, receiver_model, gamma)
    util = user_utility_from_response(theta, beta, receiver_model)
    kept = sum(int(beta.get(v, 0)) for v in (x for g in q_groups for x in g)) if receiver_model == "threshold" else None
    return float(util), kept


def run_scalability_time_vs_num_attrs_logging(
        df, candidate_cols, *, max_k=None, total_timeout_s=600.0,
        csv_path="../results/tables/time_vs_num_attrs.csv",
        plot_path="../results/plots/time_vs_num_attrs.png",
        append=False, receiver_model="threshold", bias_seed=123, print_progress=True
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    header = ["timestamp", "k_attrs", "domain_size", "time_alg1_first_s", "num_edges_alg1", "time_alg24_s",
              "utility_q_star"]
    if not append or not os.path.exists(csv_path):
        with open(csv_path, "w") as f: f.write(",".join(header) + "\n")

    start_all = time.perf_counter()
    prior = PriorSpec(kind="uniform")
    ordered_cols = pick_columns_smallest_domains(df, candidate_cols, len(candidate_cols))
    max_k = max_k or len(ordered_cols)

    xs, ys_alg24 = [], []
    for k in range(1, max_k + 1):
        cols_k = ordered_cols[:k]
        dom = build_cartesian_domain(df, cols_k)
        ksize = len(dom)
        biases = biases_from_bias_obj(dom, make_random_L_level_bias(dom, L=ksize, seed=bias_seed))

        # Alg 1 first pass
        t1a = time.perf_counter()
        C0 = algorithm_1_credibility_detection([[v] for v in dom], dom, prior, biases=biases,
                                               receiver_model=receiver_model)
        time_alg1 = time.perf_counter() - t1a

        # Alg 2 + 4
        t_start = time.perf_counter()
        q_base = algorithm_2_build_qbase(dom, dom, prior, biases=biases, receiver_model=receiver_model)
        q_star = algorithm_4_maximally_informative(q_base, dom, prior, biases=biases, receiver_model=receiver_model)
        time_alg24 = time.perf_counter() - t_start

        u_star, _ = evaluate_plan_utility(q_star, dom, prior, biases=biases, receiver_model=receiver_model)
        u_babble, _ = evaluate_plan_utility([dom[:]], dom, prior, biases=biases, receiver_model=receiver_model)
        u_base, _ = evaluate_plan_utility(q_base, dom, prior, biases=biases, receiver_model=receiver_model)

        if print_progress:
            print(f"[k={k}] cols={cols_k} => |domain|={ksize:,}")
            print(
                f"[k={k}] | Alg1_first={time_alg1:.3f}s (|C|={len(C0)}) | 2+4={time_alg24:.3f}s | q_base={len(q_base)}→q*={len(q_star)} | U(babble)={u_babble:.3f}, U(base)={u_base:.3f}, U(q★)={u_star:.3f}")

        with open(csv_path, "a") as f:
            f.write(f"{time.ctime()},{k},{ksize},{time_alg1:.4f},{len(C0)},{time_alg24:.4f},{u_star:.4f}\n")

        xs.append(k);
        ys_alg24.append(time_alg24)
        if (time.perf_counter() - start_all) > total_timeout_s: break

    if xs:
        plt.figure(figsize=(5, 3))
        plt.plot(xs, [y * 1000 for y in ys_alg24], marker="o", label="MI Time")
        plt.xlabel("Attrs");
        plt.ylabel("Time (ms)");
        plt.legend();
        plt.savefig(plot_path);
        plt.close()


def load_compas_df():
    path = "../data/real/compas_bucketized.csv"
    if os.path.exists(path): return pd.read_csv(path), True
    n = 1000;
    rng = np.random.default_rng(123)
    return pd.DataFrame({"age_cat": rng.integers(0, 3, n), "priors_count": rng.integers(0, 10, n),
                         "v_decile_score": rng.integers(0, 10, n)}), False


if __name__ == "__main__":
    df, _ = load_compas_df()
    # Support both models
    for model in ["threshold", "quadratic"]:
        print(f"\n--- Running Scalability for {model.upper()} receiver ---")
        run_scalability_time_vs_num_attrs_logging(
            df, ["age_cat", "priors_count", "v_decile_score"],
            receiver_model=model,
            csv_path=f"../results/tables/compas_{model}_3_time_vs_num_attrs.csv",
            plot_path=f"../results/plots/compas_{model}_3_time_vs_num_attrs.png"
        )