# binned_coi.py
# -*- coding: utf-8 -*-
"""
Conservative COI with time-bounded bin refinement (uniform prior + threshold).

What this does
--------------
- Uses a conservative bin-level envelope check for credibility (Alg.1@bins) with **no false positives**.
- Merges unsupported boundaries (Alg.2) on the bin partition only (fast).
- Runs the maximally-informative DP (Alg.4) on that partition.
- Evaluates utilities on the FULL item domain with fixed item-level biases.
- Utilities and kept are **weighted by original multiplicities** of distinct values.
- Increases the number of bins along a schedule (or explicit list) within a time budget.
- Logs:
    * time_alg1_s (bin-level credibility detection timing),
    * time_alg2_s, time_alg4_s,
    * utilities (babble/base/star),
    * kept counts (weighted),
    * q_base_groups, q_star_groups.

CSV columns
-----------
kind, n, bins_count, time_alg1_s, time_alg2_s, time_alg4_s,
q_base_groups, q_star_groups,
utility_babble, utility_base, utility_star,
kept_babble, kept_base, kept_star,
bias_kind, bias_direction, low, high, jitter, seed, elapsed_s
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Any
import math, argparse, csv, time, os
import numpy as np
import pandas as pd

# ---------------------- Globals ----------------------
EPS_COMPARE = 1e-9

# ---------------------- Prior ------------------------
@dataclass
class PriorSpec:
    kind: str = "uniform"
    def rank_expectation(self, k: int, r_desc: int) -> float:
        """E[Θ] for DESC rank r_desc among k under a uniform prior on permutations."""
        if k <= 0 or r_desc <= 0 or r_desc > k:
            return 0.0
        j = k - r_desc + 1  # ASC rank
        return j / (k + 1.0)

def compute_expected_posteriors(q_groups: List[List[Any]], domain: List[Any], prior: PriorSpec) -> Dict[Any, float]:
    """Group-average expected posteriors, uniform-permutation prior."""
    k = len(domain)
    post: Dict[Any, float] = {}
    r = 1
    for g in q_groups:
        n = len(g)
        if n <= 0:
            continue
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a = ((j_hi + j_lo) / 2.0) / (k + 1.0)
        for v in g:
            post[v] = float(a)
        r += n
    for v in domain:
        post.setdefault(v, 0.0)
    return post

# ---------------------- Receiver & utility ------------
def system_best_response_threshold(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    biases: Dict[Any,float]
) -> Dict[Any,int]:
    """Threshold receiver: keep(v) = 1{post[v] > bias[v]}."""
    post = compute_expected_posteriors(q_groups, domain, prior)
    return {v: 1 if (post.get(v,0.0) - biases.get(v,0.0)) > EPS_COMPARE else 0
            for g in q_groups for v in g}

def evaluate_plan_utility(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    biases: Dict[Any,float],
    *,
    counts: Optional[Dict[Any,int]] = None
) -> Tuple[float,int]:
    """
    Weighted utility/kept if counts provided:
      util = sum_v theta[v] * keep[v] * counts[v]
      kept = sum_v keep[v]              * counts[v]
    """
    if counts is None:
        counts = {v: 1 for v in domain}
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta  = system_best_response_threshold(q_groups, domain, prior, biases)
    util  = sum(theta.get(v,0.0) * float(beta.get(v,0)) * int(counts.get(v,1)) for v in domain)
    kept  = sum(int(beta.get(v,0)) * int(counts.get(v,1)) for v in domain)
    return float(util), kept

# ---------------------- FULL Alg.1 (optional baseline) -----
def algorithm_1_credibility_detection_fast_uniform_threshold(
    q_groups: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any,float]
) -> Set[Tuple[Any,Any]]:
    """
    Fast pairwise credibility edges for uniform+threshold; used only for small full baseline.
    Returns edges (u,v) considered 'credible' under Alg.1 rule (paper version).
    """
    k = len(domain)
    # group-average posteriors (DESC)
    a_g = []
    r = 1
    for g in q_groups:
        n = len(g)
        j_hi = k - r + 1
        j_lo = k - (r + n - 1) + 1
        a_g.append(((j_hi + j_lo) / 2.0) / (k + 1.0))
        r += n

    C: Set[Tuple[Any,Any]] = set()
    for gi in range(len(q_groups)):
        for gj in range(gi+1, len(q_groups)):
            ai, aj = a_g[gi], a_g[gj]   # ai > aj (DESC)
            # ambiguous interval I = [aj, ai)
            for u in q_groups[gi]:
                bu = biases[u]
                du = -1 if (bu + EPS_COMPARE >= aj and bu < ai - EPS_COMPARE) else 0
                for v in q_groups[gj]:
                    bv = biases[v]
                    dv =  1 if (bv + EPS_COMPARE >= aj and bv < ai - EPS_COMPARE) else 0
                    # non-credible only if du=0 and dv=1 (both types prefer swap)
                    if not (du == 0 and dv == 1):
                        C.add((u, v))
    return C

# ---------------------- Alg.2 (full baseline) helpers -----
def _reachability_bitsets(items: List[Any], edges: Set[Tuple[Any,Any]]):
    n = len(items)
    idx = {v:i for i,v in enumerate(items)}
    neigh = [[] for _ in range(n)]
    for u,v in edges:
        iu,iv = idx[u], idx[v]
        if iu < iv:
            neigh[iu].append(iv)
    R = [0]*n
    for i in range(n-1,-1,-1):
        mask = 0
        for j in neigh[i]:
            mask |= (1<<j) | R[j]
        R[i] = mask
    return idx, R

def _boundary_supported_by_edges(idx_map, R, Gi: List[Any], Gj: List[Any], *, policy: str = "all", alpha: float = 0.5) -> bool:
    if not Gi or not Gj:
        return True
    want_mask = 0
    for v in Gj:
        want_mask |= (1<<idx_map[v])
    if policy == "all":
        for u in Gi:
            if (R[idx_map[u]] & want_mask) != want_mask:
                return False
        return True
    # quantile
    tot = len(Gi)*len(Gj)
    sup = 0
    for u in Gi:
        mu = R[idx_map[u]]
        for v in Gj:
            if (mu >> idx_map[v]) & 1:
                sup += 1
    return sup >= math.ceil(alpha * tot)

def algorithm_2_build_qbase_full(
    initial_order: List[Any],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any,float],
    boundary_policy="all",
    alpha=0.5
) -> List[List[Any]]:
    q_cur = [[x] for x in initial_order]
    for _ in range(len(domain)+1):
        C = algorithm_1_credibility_detection_fast_uniform_threshold(q_cur, domain, prior, biases=biases)
        idx_map, R = _reachability_bitsets([x for g in q_cur for x in g], C)
        merged = False
        i = 0
        while i < len(q_cur)-1:
            Gi, Gj = q_cur[i], q_cur[i+1]
            if not _boundary_supported_by_edges(idx_map, R, Gi, Gj, policy=boundary_policy, alpha=alpha):
                q_cur = q_cur[:i] + [Gi+Gj] + q_cur[i+2:]
                merged = True
                break
            i += 1
        if not merged:
            break
    return q_cur

# ---------------------- Alg.4 (DP; weighted) ----------------
def algorithm_4_maximally_informative(
    q_base: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any,float],
    counts: Optional[Dict[Any,int]] = None
) -> List[List[Any]]:
    """
    DP scoring uses weighted utilities if 'counts' provided.
    """
    if counts is None:
        counts = {v: 1 for v in domain}
    m = len(q_base)
    if m == 0:
        return []
    theta = compute_expected_posteriors([[v] for v in domain], domain, prior)
    beta_base = system_best_response_threshold(q_base, domain, prior, biases)
    base_gain = []
    for t in range(m):
        items_t = q_base[t]
        u0 = sum(theta.get(v,0.0) * float(beta_base.get(v,0)) * int(counts.get(v,1)) for v in items_t)
        base_gain.append(u0)
    C0 = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(i,m):
            run_items = [x for g in q_base[i:j+1] for x in g]
            temp_q = q_base[:i] + [run_items] + q_base[j+1:]
            beta_run = system_best_response_threshold(temp_q, domain, prior, biases)
            u_run = sum(theta.get(v,0.0) * float(beta_run.get(v,0)) * int(counts.get(v,1)) for v in run_items)
            base_line = sum(base_gain[t] for t in range(i,j+1))
            C0[i][j] = u_run - base_line
    if max(C0[i][j] for i in range(m) for j in range(i,m)) <= EPS_COMPARE:
        return q_base
    dp = [0.0]*(m+1)
    prev = [-1]*(m+1)
    for t in range(1,m+1):
        best, arg = -1e18, -1
        for i in range(1,t+1):
            g0 = C0[i-1][t-1]
            val = dp[i-1] + (g0 if g0 > EPS_COMPARE else 0.0)
            if (val > best + EPS_COMPARE) or (abs(val - best) <= EPS_COMPARE and (i-1) > arg):
                best, arg = val, i-1
        dp[t], prev[t] = best, arg
    q_star: List[List[Any]] = []
    t = m
    while t > 0:
        i = prev[t]
        q_star.insert(0, [x for g in q_base[i:t] for x in g])
        t = i
    return q_star

# ---------------------- Conservative bins (sound) -----
def _bin_boundary_supported_envelope(Gi: List[Any], Gj: List[Any], *, idx: Dict[Any,int], k: int, prior: PriorSpec, biases: Dict[Any,float]) -> bool:
    """
    Envelope sufficient condition: if boundary is 'supported' at bin level, all item pairs crossing it
    are credible (no false positives). If found unsupported, we conservatively MERGE (Alg.2).
    """
    # Worst-case group posteriors for bins (DESC ranks are 1..k)
    i_lo = max(idx[x] for x in Gi) + 1
    j_hi = min(idx[x] for x in Gj) + 1
    theta_i_low  = prior.rank_expectation(k, i_lo)   # lowest posterior inside Gi
    theta_j_high = prior.rank_expectation(k, j_hi)   # highest posterior inside Gj

    # If ranges overlap, boundary is safe (supported).
    if theta_i_low <= theta_j_high + EPS_COMPARE:
        return True

    # Bias envelopes for bins
    bi_min = min(biases[x] for x in Gi); bi_max = max(biases[x] for x in Gi)
    bj_min = min(biases[x] for x in Gj); bj_max = max(biases[x] for x in Gj)

    # Sufficient "safety" conditions (conservative):
    vsafe = (bj_max < theta_j_high - EPS_COMPARE) or (bj_min >= theta_i_low + EPS_COMPARE)
    usafe = (bi_min >= theta_j_high + EPS_COMPARE) and (bi_max < theta_i_low - EPS_COMPARE)
    return vsafe or usafe

def algorithm_2_build_qbase_conservative_bins(
    q_bins: List[List[Any]],
    domain: List[Any],
    prior: PriorSpec,
    *,
    biases: Dict[Any,float]
) -> List[List[Any]]:
    q_cur = [g[:] for g in q_bins]
    idx = {v:i for i,v in enumerate(domain)}
    k = len(domain)
    while True:
        merged = False
        for i in range(len(q_cur)-1):
            Gi, Gj = q_cur[i], q_cur[i+1]
            supported = _bin_boundary_supported_envelope(Gi, Gj, idx=idx, k=k, prior=prior, biases=biases)
            if not supported:
                q_cur = q_cur[:i] + [Gi+Gj] + q_cur[i+2:]
                merged = True
                break
        if not merged:
            return q_cur

# ---------------------- First CD timer on bins --------
def time_binlevel_CD(bins: List[List[Any]], domain: List[Any], prior: PriorSpec, biases: Dict[Any,float]) -> float:
    """Time a single pass of bin-level credibility checks (envelope test) across boundaries."""
    idx = {v:i for i,v in enumerate(domain)}
    k = len(domain)
    t0 = time.perf_counter()
    for i in range(len(bins)-1):
        _ = _bin_boundary_supported_envelope(bins[i], bins[i+1], idx=idx, k=k, prior=prior, biases=biases)
    return time.perf_counter() - t0

# ---------------------- Bias generators ---------------
def _rank_percentiles_desc(domain: List[Any]) -> Dict[Any,float]:
    k = len(domain)
    return {v: (k - (i + 0.5)) / k for i, v in enumerate(domain)}
def make_bias_map(
    domain: List[Any],
    *,
    kind: str = "random_multilevel",     # "sigmoid" | "quantile_steps" | "power" | "hockey" | "random_multilevel"
    direction: str = "down",
    low: float = 0.1,
    high: float = 0.9,
    center: float = 0.65,
    ksig: float = 12.0,
    steps: int = 10,
    alpha: float = 1.3,
    tau: float = 0.7,
    levels = (0.8, 0.6, 0.3, 0.4),       # NEW: can also be an int (number of levels)
    probs:  Optional[Tuple[float, ...]] = (0.25, 0.25, 0.25, 0.25),  # NEW: can be None for uniform over L
    jitter: float = 0.0,
    seed: int = 123,
) -> Dict[Any,float]:
    """
    Default remains the same (4 levels with equal probs).
    You can now pass:
      - levels = int (e.g., len(domain)//2) to auto-generate L evenly-spaced levels in (0.05, 0.95)
      - probs  = None to use uniform over those L levels
      - or keep tuple behaviors exactly as before.
    """
    def _clip(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    rng = np.random.default_rng(seed)
    p = _rank_percentiles_desc(domain)  # used by shaped options
    out: Dict[Any,float] = {}

    if kind == "random_multilevel":
        # ---- accept int or tuple/list/ndarray for 'levels'
        if isinstance(levels, int):
            L = max(2, int(levels))
            lvl = np.linspace(0.45, 0.95, num=L)  # avoid exact 0/1 edges
            levels_tuple = tuple(float(x) for x in lvl)
        elif isinstance(levels, (list, tuple, np.ndarray)):
            levels_tuple = tuple(float(x) for x in levels)
            L = len(levels_tuple)
            if L < 2:
                raise ValueError("levels must have at least 2 distinct values.")
        else:
            # fallback to original default if something odd is passed
            levels_tuple = (0.8, 0.6, 0.3, 0.4)
            L = len(levels_tuple)

        # ---- probs handling: None => uniform; else must match L
        if probs is None:
            P = np.ones(L, dtype=float) / float(L)
        else:
            P = np.array(probs, dtype=float)
            if len(P) != L:
                raise ValueError(f"'probs' length ({len(P)}) must match number of levels ({L}).")
            S = P.sum()
            if S <= 0:
                raise ValueError("'probs' must sum to a positive value.")
            P = P / S

        labels = rng.choice(L, size=len(domain), p=P)
        for i, v in enumerate(domain):
            base = levels_tuple[int(labels[i])]
            if jitter > 0:
                base += float(rng.uniform(-jitter, jitter))
            out[v] = _clip(float(base))
        return out

    # ---- shaped options unchanged
    for v in domain:
        pv = float(p[v])
        if kind == "sigmoid":
            s = 1.0 / (1.0 + math.exp(-ksig * (pv - center)))
            base = low + (high - low) * (1.0 - s if direction == "down" else s)
        elif kind == "quantile_steps":
            q_idx = min(steps - 1, int(math.floor(pv * steps)))
            t = q_idx / max(1, steps - 1)
            base = (low + (high - low) * (1.0 - t)) if direction == "down" else (low + (high - low) * t)
        elif kind == "power":
            val = pv ** alpha
            base = low + (high - low) * (1.0 - val if direction == "down" else val)
        elif kind == "hockey":
            base = (low if pv >= tau else high) if direction == "down" else (high if pv >= tau else low)
        else:
            raise ValueError(f"Unknown bias kind: {kind}")
        if jitter > 0:
            base += float(rng.uniform(-jitter, jitter))
        out[v] = _clip(float(base))
    return out


# ---------------------- Binning helpers ---------------
def build_equal_bins_desc(domain: List[Any], B: int) -> List[List[Any]]:
    """Split DESC-ordered domain into B contiguous bins (as equal in size as possible)."""
    n = len(domain)
    if B <= 1:
        return [domain[:]]
    bins: List[List[Any]] = []
    for b in range(B):
        s = (b * n) // B
        e = ((b + 1) * n) // B
        if s < e:
            bins.append(domain[s:e])
    return bins

def next_bins(B: int, n: int, *, mode: str = "geometric", factor: float = 1.5) -> int:
    """Schedule next bin count."""
    if B >= n:
        return n
    if mode == "linear":
        return min(n, B + max(1, n // 50))  # +2% of n per step (>=1)
    # geometric default
    return min(n, max(B + 1, int(math.ceil(B * factor))))

# ---------------------- Data loading ------------------
def load_column_domain_with_counts(csv_path: str, col: str) -> Tuple[List[Any], Dict[Any,int]]:
    """
    Returns (distinct DESC domain, counts per distinct value) for a numeric column.
    """
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {csv_path}")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    vc = vals.value_counts()  # index: value, value: count
    domain = sorted(vc.index.tolist(), reverse=True)  # DESC numeric order
    # store as float keys to match posteriors map if needed
    counts = {float(v): int(vc.loc[v]) for v in vc.index}
    # Normalize keys: ensure domain items are the same objects used in counts
    domain = [float(v) for v in domain]
    return domain, counts

# ---------------------- Experiment (with timeout) -----
def run_experiment_with_timeout(
    domain: List[Any],
    *,
    counts: Optional[Dict[Any,int]] = None,
    time_budget_s: float = 600.0,
    start_bins: int = 2,
    growth_mode: str = "geometric",
    growth_factor: float = 1.5,
    explicit_bins: Optional[List[int]] = None,
    bias_kind: str = "random_multilevel",
    bias_direction: str = "down",
    low: float = 0.15, high: float = 0.85,
    center: float = 0.65, ksig: float = 10.0,
    steps: int = 12, alpha: float = 1.4, tau: float = 0.7,
    jitter: float = 0.0,
    bias_seed: int = 123,
    with_full: bool = False,
    csv_path: str = "coi_binning_results.csv",
    append: bool = False,
    print_progress: bool = True,
    # ---- NEW controls for multi-run:
    runs: int = 5,
    resample_bias_each_run: bool = True,
    # ---- Existing optional bias knobs (kept for compatibility):
    levels: Optional[Any] = None,                       # int or tuple; None => default behavior
    probs: Optional[Tuple[float, ...]] = None           # None => uniform over L if levels is int
):
    """
    Runs the experiment for each bins_count in the schedule, repeating it `runs` times.
    Adds a 'run' column to the CSV. If resample_bias_each_run is True, biases are
    regenerated each run using (bias_seed + run_idx) for reproducibility.
    """
    n = len(domain)
    prior = PriorSpec(kind="uniform")
    if counts is None:
        counts = {v: 1 for v in domain}

    # CSV header
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if (not append) and os.path.exists(csv_path):
        os.remove(csv_path)
    want_header = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    if want_header:
        w.writerow([
            "run",                      # <-- NEW
            "kind","n","bins_count",
            "time_alg1_s","time_alg2_s","time_alg4_s",
            "q_base_groups","q_star_groups",
            "utility_babble","utility_base","utility_star",
            "kept_babble","kept_base","kept_star",
            "bias_kind","bias_direction","low","high","jitter","seed","elapsed_s"
        ]); f.flush()

    # Bin schedule under time budget (unchanged)
    if explicit_bins:
        schedule = sorted({min(n, max(2, b)) for b in explicit_bins})
    else:
        schedule = []
        B = min(n, max(2, start_bins))
        seen = set()
        while B not in seen:
            schedule.append(B)
            seen.add(B)
            if B >= n:
                break
            B = next_bins(B, n, mode=growth_mode, factor=growth_factor)

    start_all = time.perf_counter()

    # === Outer loop over runs ===
    for run_idx in range(1, max(1, runs) + 1):
        # Biases per run (either re-sampled or fixed across all runs)
        local_seed = (bias_seed + run_idx) if resample_bias_each_run else bias_seed

        if bias_kind == "random_multilevel":
            # Respect optional levels/probs if provided; otherwise keep your default behavior
            if levels is None:
                biases = make_bias_map(
                    domain,
                    kind="random_multilevel",
                    levels=(len(domain) // 2),  # your current default tweak
                    probs=None,
                    jitter=jitter,
                    seed=local_seed
                )
            else:
                biases = make_bias_map(
                    domain,
                    kind="random_multilevel",
                    levels=levels,
                    probs=probs,
                    jitter=jitter,
                    seed=local_seed
                )
        else:
            biases = make_bias_map(
                domain,
                kind=bias_kind, direction=bias_direction,
                low=low, high=high,
                center=center, ksig=ksig,
                steps=steps, alpha=alpha, tau=tau,
                jitter=jitter, seed=local_seed
            )

        # Babble baseline (anchor) — WEIGHTED, logged once per run with run=0 or run_idx?
        # We'll record it per run with bins_count=1 and run=run_idx for clarity.
        util_bab, kept_bab = evaluate_plan_utility([domain[:]], domain, prior, biases, counts=counts)
        elapsed0 = time.perf_counter() - start_all
        w.writerow([run_idx, "babble", n, 1, 0.0, 0.0, 0.0,
                    1, 1,
                    util_bab, util_bab, util_bab,
                    kept_bab, kept_bab, kept_bab,
                    bias_kind, bias_direction, low, high, jitter, local_seed,
                    elapsed0]); f.flush()
        if print_progress:
            print(f"[run {run_idx} | babble] n={n} | U={util_bab:.4f} | kept={kept_bab}")

        # === Inner loop over bins in the schedule ===
        for B in schedule:
            # Time budget check
            if (time.perf_counter() - start_all) >= time_budget_s or B > 1600:
                if print_progress:
                    print(f"⏱️  Time budget reached ({time_budget_s}s). Stopping at bins={B}, run={run_idx}.")
                break

            bins = build_equal_bins_desc(domain, B)

            # --- Alg.1 on bins
            time_alg1 = time_binlevel_CD(bins, domain, prior, biases)

            # --- Alg.2 on bins
            t2a = time.perf_counter()
            q_base = algorithm_2_build_qbase_conservative_bins(bins, domain, prior, biases=biases)
            t2b = time.perf_counter()

            # --- Alg.4 DP
            t4a = time.perf_counter()
            q_star = algorithm_4_maximally_informative(q_base, domain, prior, biases=biases, counts=counts)
            t4b = time.perf_counter()

            util_base, kept_base = evaluate_plan_utility(q_base, domain, prior, biases, counts=counts)
            util_star, kept_star = evaluate_plan_utility(q_star, domain, prior, biases, counts=counts)

            elapsed = time.perf_counter() - start_all
            w.writerow([run_idx, "binned", n, B, time_alg1, (t2b - t2a), (t4b - t4a),
                        len(q_base), len(q_star),
                        util_bab, util_base, util_star,
                        kept_bab, kept_base, kept_star,
                        bias_kind, bias_direction, low, high, jitter, local_seed,
                        elapsed]); f.flush()

            if print_progress:
                print(f"[run {run_idx} | bins={B:5d}] q_base={len(q_base):4d}→q*={len(q_star):4d} | "
                      f"U(babble)={util_bab:.4f}, U(base)={util_base:.4f}, U(q*)={util_star:.4f} | "
                      f"kept(babble)={kept_bab}, kept(base)={kept_base}, kept(q*)={kept_star} | "
                      f"t1={time_alg1:.4f}s t2={(t2b-t2a):.4f}s t4={(t4b-t4a):.4f}s | elapsed={elapsed:.1f}s")

        # Optional FULL baseline (kept intact; small n only)
        if with_full and n <= 4:
            q_singletons = [[v] for v in domain]
            t1a = time.perf_counter()
            _ = algorithm_1_credibility_detection_fast_uniform_threshold(q_singletons, domain, prior, biases=biases)
            t1b = time.perf_counter()
            time_alg1_full = (t1b - t1a)

            t2a = time.perf_counter()
            q_base_full = algorithm_2_build_qbase_full(domain, domain, prior, biases=biases, boundary_policy="all", alpha=0.5)
            t2b = time.perf_counter()

            t4a = time.perf_counter()
            q_star_full = algorithm_4_maximally_informative(q_base_full, domain, prior, biases=biases, counts=counts)
            t4b = time.perf_counter()

            util_base_full, kept_base_full = evaluate_plan_utility(q_base_full, domain, prior, biases, counts=counts)
            util_star_full, kept_star_full = evaluate_plan_utility(q_star_full, domain, prior, biases, counts=counts)

            elapsed = time.perf_counter() - start_all
            w.writerow([run_idx, "full", n, n, time_alg1_full, (t2b - t2a), (t4b - t4a),
                        len(q_base_full), len(q_star_full),
                        util_bab, util_base_full, util_star_full,
                        kept_bab, kept_base_full, kept_star_full,
                        bias_kind, bias_direction, low, high, jitter, local_seed,
                        elapsed]); f.flush()
            if print_progress:
                print(f"[run {run_idx} | full] n={n} | q_base={len(q_base_full)}→q*={len(q_star_full)} | "
                      f"U(babble)={util_bab:.4f}, U(base)={util_base_full:.4f}, U(q*)={util_star_full:.4f} | "
                      f"kept(babble)={kept_bab}, kept(base)={kept_base_full}, kept(q*)={kept_star_full} | "
                      f"t1={time_alg1_full:.4f}s t2={t2b-t2a:.4f}s t4={t4b-t4a:.4f}s | elapsed={elapsed:.1f}s")

    f.close()
    print("CSV ->", csv_path)



# ---------------------- CLI ---------------------------
def parse_int_list(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_csv", type=str, default="", help="Path to CSV with the numeric column (use --col).")
    ap.add_argument("--col", type=str, default="price", help="Numeric column to build the distinct domain from.")
    ap.add_argument("--n", type=int, default=0, help="Domain size if not loading from CSV (DESC integers).")
    ap.add_argument("--bins", type=str, default="", help="Explicit bin counts (comma-separated). If empty, use growth schedule.")
    ap.add_argument("--time_budget_s", type=float, default=600.0, help="Global time budget (seconds) per dataset run.")
    ap.add_argument("--start_bins", type=int, default=2)
    ap.add_argument("--growth_mode", type=str, default="geometric", choices=["geometric","linear"])
    ap.add_argument("--growth_factor", type=float, default=1.5)
    ap.add_argument("--with_full", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="Append to CSV instead of overwriting.")

    # Bias controls
    ap.add_argument("--bias_kind", type=str, default="random_multilevel",
                    choices=["sigmoid","quantile_steps","power","hockey","random_multilevel"])
    ap.add_argument("--bias_dir",  type=str, default="down", choices=["down","up"])
    ap.add_argument("--low",   type=float, default=0.15)
    ap.add_argument("--high",  type=float, default=0.85)
    ap.add_argument("--center",type=float, default=0.65)
    ap.add_argument("--ksig",  type=float, default=10.0)
    ap.add_argument("--steps", type=int,   default=12)
    ap.add_argument("--alpha", type=float, default=1.4)
    ap.add_argument("--tau",   type=float, default=0.7)
    ap.add_argument("--jitter",type=float, default=0.00)
    ap.add_argument("--seed",  type=int,   default=123)

    # Optional random_multilevel knobs (kept for compatibility)
    ap.add_argument("--levels", type=str, default="",
                    help="For random_multilevel: int (L) or comma-separated float levels. Empty = default.")
    ap.add_argument("--probs", type=str, default="",
                    help="For random_multilevel: comma-separated probabilities matching levels. Empty = uniform for int levels, or default for tuple levels.")

    # Multi-run controls
    ap.add_argument("--runs", type=int, default=5, help="Number of runs per bins_count.")
    ap.add_argument("--resample_bias_each_run", type=int, default=1,
                    help="1 to resample biases with seed+run each run; 0 to keep same biases across runs.")

    ap.add_argument("--csv_out",   type=str,   default="coi_binning_results.csv")
    args = ap.parse_args()

    # Domain (+ optional counts)
    if args.from_csv.strip():
        domain, counts = load_column_domain_with_counts(args.from_csv, args.col)
    else:
        if args.n <= 0:
            raise ValueError("Provide --from_csv (with --col) OR a positive --n.")
        domain = list(range(args.n, 0, -1))
        counts = {v: 1 for v in domain}

    # Parse optional levels/probs for random_multilevel
    levels_opt: Optional[Any] = None
    probs_opt: Optional[Tuple[float, ...]] = None
    if args.levels.strip():
        tok = args.levels.strip()
        if tok.isdigit():
            levels_opt = int(tok)
        else:
            levels_opt = tuple(float(x) for x in tok.split(",") if x.strip())
    if args.probs.strip():
        probs_opt = tuple(float(x) for x in args.probs.split(",") if x.strip())

    # Bin schedule (optional explicit)
    explicit_bins = parse_int_list(args.bins) if args.bins.strip() else None

    run_experiment_with_timeout(
        domain,
        counts=counts,
        time_budget_s=args.time_budget_s,
        start_bins=args.start_bins,
        growth_mode=args.growth_mode,
        growth_factor=args.growth_factor,
        explicit_bins=explicit_bins,
        bias_kind=args.bias_kind,
        bias_direction=args.bias_dir,
        low=args.low, high=args.high,
        center=args.center, ksig=args.ksig,
        steps=args.steps, alpha=args.alpha, tau=args.tau,
        jitter=args.jitter, bias_seed=args.seed,
        with_full=bool(args.with_full),
        csv_path=args.csv_out,
        append=bool(args.append),
        print_progress=True,
        runs=max(1, args.runs),
        resample_bias_each_run=bool(args.resample_bias_each_run),
        levels=levels_opt,
        probs=probs_opt
    )


if __name__ == "__main__":
    main()
