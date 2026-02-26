"""
Comparative Cheap Talk – Group-level Credibility (paper notation)
=================================================================

You said:
1) Intent order τ is ALWAYS DESC on one column (already bucketized in your file).
2) Create disjoint groups e (distinct values of that column) and choose z groups => E, with z = |E|.
3) Randomly assign bias to groups: b(e) from m discrete bias levels.
4) β(I) is RANDOM: pick a random top-k subset of E (not necessarily consistent with τ).
5) Output non-credible AND credible returned groups.

This script implements exactly that, using your notation:

- E : set of groups (we treat each group as a "tuple" e in the paper)
- z = |E|
- Δ = {-(z-1), …, z-1} : possible differences between two ranks in [z]
- H(δ)  = {(r,r') in [z]^2 : r - r' >= δ}
- H'(δ) = complement on the integer lattice, i.e. r - r' <= δ-1
- Φ_e(H) : posterior mean rank of e under uniform prior on [z]^2 conditioned on H
- β_Φ, β'_Φ : DS best-response interpretations after posteriors induced by H and H'
- Π_[z] : projection to {1,…,z} (nearest integer, clipped)
- (δ^-_{b,b'}, δ^+_{b,b'}) : extremal feasible indifference thresholds for each ordered bias pair (b,b')

FAST part:
-----------
We DO NOT scan all z^2 points to check user incentives.
For quadratic U^r, the difference U^r(·,β) - U^r(·,β') is linear in (r,r'),
so it suffices to check the inequality at a constant number of polygon vertices.
This makes threshold computation ~ O(B^2 * z) instead of O(B^2 * z^3),
where B = number of distinct bias values.

You can run this on any CSV + any already-bucketized column.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0) Utilities: DISTINCT data, parsing group values, projection Π_[z]
# =============================================================================

def make_distinct_df(df: pd.DataFrame) -> pd.DataFrame:
    """(1) Make dataset DISTINCT."""
    return df.drop_duplicates().reset_index(drop=True)


def _is_numeric_like(series: pd.Series, frac_ok: float = 0.98) -> Tuple[bool, pd.Series]:
    """
    Decide if group labels are numeric-like (handles ints stored as strings).
    Returns (is_numeric_like, numeric_converted_series).
    """
    s0 = series.dropna()
    if len(s0) == 0:
        return False, s0
    sn = pd.to_numeric(s0, errors="coerce")
    ok = float(sn.notna().mean()) >= frac_ok
    return ok, sn


def Pi_rank(x: float, z: int) -> int:
    """Π_[z](x): nearest integer in {1,...,z}, clipped. 0.5 rounds up."""
    return int(np.clip(np.floor(x + 0.5), 1, z))


# =============================================================================
# 1) Intent τ is ALWAYS DESC: build E and ranks r(e, τ(E))
# =============================================================================

def build_groups_desc(df: pd.DataFrame, intent_col: str) -> List[Any]:
    """
    Groups are distinct values of the intent column.
    τ is ORDER BY intent_col DESC => groups sorted descending.
    """
    s_raw = df[intent_col].dropna()
    if len(s_raw) == 0:
        raise ValueError(f"Column '{intent_col}' has no non-null values.")

    is_num, s_num = _is_numeric_like(s_raw)

    if is_num:
        vals = sorted(set(s_num.dropna().tolist()), key=float, reverse=True)
        return vals
    else:
        vals = sorted(set(s_raw.astype(str).tolist()), key=str, reverse=True)
        return vals


def choose_E(groups_desc: List[Any], z: int, seed: int = 0) -> List[Any]:
    """
    Choose E as exactly z groups, uniformly at random from all groups,
    then keep DESC order (τ restricted to E).
    """
    if z <= 0:
        raise ValueError("z must be positive.")
    if z >= len(groups_desc):
        return list(groups_desc)

    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(groups_desc, size=z, replace=False).tolist())
    return [g for g in groups_desc if g in chosen]  # preserve DESC order


def rank_in_tau_E(E_desc: List[Any]) -> Dict[Any, int]:
    """Compute r(e, τ(E)) as 1..z for E in DESC intent order."""
    return {e: i + 1 for i, e in enumerate(E_desc)}


# =============================================================================
# 2) Random bias on groups: b(e) from m levels
# =============================================================================

def make_bias_levels(m: int, min_bias: float, max_bias: float, include_zero: bool = True) -> List[float]:
    """
    Create m bias levels (e.g., [0, 0.5, 1.0, 1.5, 2.0]).
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    levels = np.linspace(float(min_bias), float(max_bias), m).tolist()
    if include_zero and m >= 2:
        # ensure 0 is included (replace closest level by 0)
        idx = int(np.argmin([abs(x - 0.0) for x in levels]))
        levels[idx] = 0.0
    # keep distinct and sorted
    levels = sorted(set(float(x) for x in levels))
    return levels


def random_bias_map(E: List[Any], levels: List[float], seed: int = 0) -> Dict[Any, float]:
    """Assign each group e in E an i.i.d. random bias b(e) from levels."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.array(levels, dtype=float), size=len(E), replace=True)
    return {e: float(b) for e, b in zip(E, draws)}


# =============================================================================
# 3) Random β(I): pick random top-k subset of E
# =============================================================================

def default_k(z: int, frac: float = 0.25) -> int:
    """Default k = ceil(frac * z)."""
    return max(1, int(np.ceil(frac * z)))


def random_beta_I(E_desc: List[Any], k: int, seed: int = 0) -> List[Any]:
    """β(I) returns a random subset of size k from E."""
    rng = np.random.default_rng(seed)
    k = min(k, len(E_desc))
    return rng.choice(E_desc, size=k, replace=False).tolist()


# =============================================================================
# 4) Subroutine: Compute Indifference Thresholds (FAST)
# =============================================================================
# We compute bounds by ordered bias-pair (b,b') because Φ depends only on (z,δ)
# and the DS best-response uses Π_[z](Φ - b).
#
# Quadratic user utility implies:
#   U^r((r,r'),β) >= U^r((r,r'),β')
# is equivalent to a linear inequality in (r,r').
# Therefore, to verify it on H(δ) (a convex polygon), it suffices to check vertices.

def _vertices_halfplane_ge(z: int, c: float) -> List[Tuple[float, float]]:
    """
    Vertices of P = {(r,r') in [1,z]^2 : r - r' >= c} (continuous polygon).
    """
    pts: List[Tuple[float, float]] = []
    corners = [(1.0, 1.0), (1.0, float(z)), (float(z), 1.0), (float(z), float(z))]
    for x, y in corners:
        if x - y >= c - 1e-12:
            pts.append((x, y))

    # intersections of line r-r'=c with box edges
    y = 1.0 - c
    if 1.0 <= y <= float(z):
        pts.append((1.0, y))
    y = float(z) - c
    if 1.0 <= y <= float(z):
        pts.append((float(z), y))
    x = 1.0 + c
    if 1.0 <= x <= float(z):
        pts.append((x, 1.0))
    x = float(z) + c
    if 1.0 <= x <= float(z):
        pts.append((x, float(z)))

    # de-dup
    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9 for q in uniq):
            uniq.append(p)
    return uniq


def _vertices_halfplane_le(z: int, c: float) -> List[Tuple[float, float]]:
    """
    Vertices of Q = {(r,r') in [1,z]^2 : r - r' <= c}.
    """
    pts: List[Tuple[float, float]] = []
    corners = [(1.0, 1.0), (1.0, float(z)), (float(z), 1.0), (float(z), float(z))]
    for x, y in corners:
        if x - y <= c + 1e-12:
            pts.append((x, y))

    # intersections with r-r'=c
    y = 1.0 - c
    if 1.0 <= y <= float(z):
        pts.append((1.0, y))
    y = float(z) - c
    if 1.0 <= y <= float(z):
        pts.append((float(z), y))
    x = 1.0 + c
    if 1.0 <= x <= float(z):
        pts.append((x, 1.0))
    x = float(z) + c
    if 1.0 <= x <= float(z):
        pts.append((x, float(z)))

    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9 for q in uniq):
            uniq.append(p)
    return uniq


def _pref_linear_coeffs(beta: Tuple[int, int], beta_prime: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    For quadratic user utility:
      U^r((r,r'),beta) >= U^r((r,r'),beta')
    <=>  L1*r + L2*r' <= RHS
    where:
      L1 = 2(a' - a), L2 = 2(c' - c),
      RHS = (a'^2 - a^2) + (c'^2 - c^2)
    with beta=(a,c), beta'=(a',c').
    """
    a, c = beta
    ap, cp = beta_prime
    L1 = 2.0 * (ap - a)
    L2 = 2.0 * (cp - c)
    RHS = (ap * ap - a * a) + (cp * cp - c * c)
    return L1, L2, RHS


def _holds_on_region_ge(z: int, delta: int, beta: Tuple[int, int], beta_prime: Tuple[int, int]) -> bool:
    """
    Check U^r(·,beta) >= U^r(·,beta') for all (r,r') with r-r' >= delta.
    """
    verts = _vertices_halfplane_ge(z, float(delta))
    if not verts:
        return True  # empty region
    L1, L2, RHS = _pref_linear_coeffs(beta, beta_prime)
    # must hold everywhere => max(LHS-RHS) over region <= 0 (achieved at a vertex)
    max_violation = max(L1 * x + L2 * y - RHS for x, y in verts)
    return max_violation <= 1e-12


def _holds_on_region_le(z: int, c: int, beta: Tuple[int, int], beta_prime: Tuple[int, int]) -> bool:
    """
    Check U^r(·,beta) >= U^r(·,beta') for all (r,r') with r-r' <= c.
    (We use c = delta-1 for H'(delta) on the lattice.)
    """
    verts = _vertices_halfplane_le(z, float(c))
    if not verts:
        return True
    L1, L2, RHS = _pref_linear_coeffs(beta, beta_prime)
    max_violation = max(L1 * x + L2 * y - RHS for x, y in verts)
    return max_violation <= 1e-12


def precompute_uniform_posterior_means(z: int) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Under uniform prior on [z]^2 for integer ranks,
    for each δ in Δ compute:
      Φ_r(H), Φ_r'(H), Φ_r(H'), Φ_r'(H')
    where:
      H(δ)  = {r-r' >= δ}
      H'(δ) = complement on integer lattice = {r-r' <= δ-1}
    """
    ranks = np.arange(1, z + 1)
    r, rp = np.meshgrid(ranks, ranks, indexing="ij")

    out: Dict[int, Tuple[float, float, float, float]] = {}
    for delta in range(-(z - 1), z):
        H = (r - rp) >= delta
        Hc = (r - rp) <= (delta - 1)
        if H.sum() == 0 or Hc.sum() == 0:
            out[delta] = (np.nan, np.nan, np.nan, np.nan)
        else:
            out[delta] = (
                float(r[H].mean()), float(rp[H].mean()),
                float(r[Hc].mean()), float(rp[Hc].mean())
            )
    return out


def compute_indifference_thresholds_by_bias_pair(z: int, bias_values: Iterable[float]) -> Dict[Tuple[float, float], Optional[Tuple[int, int]]]:
    """
    Paper: Algorithm 'Compute Indifference Threshold' but computed per ordered bias-pair (b,b').

    Output:
      bounds[(b,b')] = (δ^-_{b,b'}, δ^+_{b,b'})  or None (= Fail)
    """
    bias_vals = sorted(set(float(b) for b in bias_values))
    post = precompute_uniform_posterior_means(z)

    bounds: Dict[Tuple[float, float], Optional[Tuple[int, int]]] = {}

    for b in bias_vals:
        for bp in bias_vals:
            if abs(b - bp) < 1e-12:
                # equal biases -> typically no unsafe band in this model
                bounds[(b, bp)] = (0, 0)
                continue

            found = False
            d_minus, d_plus = 0, 0

            for delta in range(-(z - 1), z):
                Phi_r_H, Phi_rp_H, Phi_r_Hc, Phi_rp_Hc = post[delta]
                if np.isnan(Phi_r_H) or np.isnan(Phi_r_Hc):
                    continue

                # DS best responses (quadratic):
                # r(e,β_Φ) = Π_[z]( Φ_e(H) - b(e) )
                beta_H  = (Pi_rank(Phi_r_H  - b,  z), Pi_rank(Phi_rp_H  - bp, z))
                beta_Hc = (Pi_rank(Phi_r_Hc - b,  z), Pi_rank(Phi_rp_Hc - bp, z))

                # influential requires different interpretations across posteriors
                if beta_H == beta_Hc:
                    continue

                # User IC checks (Definition equilibrium):
                #   all points in H prefer beta_H
                #   all points in H' prefer beta_Hc
                ok1 = _holds_on_region_ge(z, delta,     beta_H,  beta_Hc)
                ok2 = _holds_on_region_le(z, delta - 1, beta_Hc, beta_H)

                if ok1 and ok2:
                    found = True
                    if delta < 0:
                        d_minus = min(d_minus, delta)
                    elif delta > 0:
                        d_plus = max(d_plus, delta)

            bounds[(b, bp)] = (d_minus, d_plus) if found else None

    return bounds


# =============================================================================
# 5) Detect Credible Information (top-k membership test, paper-style names)
# =============================================================================

def detect_credible_and_noncredible(
    E_desc: List[Any],
    r_tau_E: Dict[Any, int],
    b_map: Dict[Any, float],
    bounds_by_bias: Dict[Tuple[float, float], Optional[Tuple[int, int]]],
    beta_I: List[Any],
) -> Dict[str, Any]:
    """
    Paper: Detect Credible Information (returns non-credible AND credible).

    For each returned e in β(I), look for a missing e' in E \ β(I) that beats e in τ(E):
       r_tau_E(e') < r_tau_E(e)
    Let gap = r_tau_E(e) - r_tau_E(e') > 0.
    Flag e as non-credible if:
      - bounds for (b(e), b(e')) is Fail, OR
      - 0 <= gap < δ^+_{b(e),b(e')}
    """
    beta_set = set(beta_I)
    non_credible_returned: List[Any] = []
    credible_returned: List[Any] = []
    witnesses: Dict[Any, Dict[str, Any]] = {}

    for e in beta_I:
        re = r_tau_E[e]
        be = float(b_map[e])
        flagged = False

        for ep in E_desc:
            if ep in beta_set:
                continue
            rep = r_tau_E[ep]
            if rep >= re:
                continue  # ep is not better under τ(E)

            bep = float(b_map[ep])
            bnd = bounds_by_bias.get((be, bep), None)

            if bnd is None:
                non_credible_returned.append(e)
                witnesses[e] = {"missing": ep, "reason": "Fail (no feasible δ) for this bias pair"}
                flagged = True
                break

            _, delta_plus = bnd
            gap = re - rep
            if 0 <= gap < delta_plus:
                non_credible_returned.append(e)
                witnesses[e] = {"missing": ep, "reason": f"gap={gap} < δ^+={delta_plus}"}
                flagged = True
                break

        if not flagged:
            credible_returned.append(e)

    return {
        "beta_I": beta_I,
        "credible_returned": credible_returned,
        "non_credible_returned": non_credible_returned,
        "witnesses": witnesses,
    }


# =============================================================================
# 6) One simple runner (with simple caching because (z, bias_levels) may repeat)
# =============================================================================

_BOUNDS_CACHE: Dict[Tuple[int, Tuple[float, ...]], Dict[Tuple[float, float], Optional[Tuple[int, int]]]] = {}


def run_one(
    csv_path: str,
    intent_col: str,
    z: int,
    m_levels: int,
    min_bias: float,
    max_bias: float,
    k_frac: float = 0.25,
    seed_groups: int = 0,
    seed_bias: int = 1,
    seed_beta: int = 2,
) -> Dict[str, Any]:
    """
    Single run:
      - τ is DESC on intent_col
      - choose E of size z
      - random bias b(e) from m levels
      - random β(I) of size k = ceil(k_frac*z)
      - compute thresholds once per (z, levels) (cache)
      - output credible + non-credible returned groups
    """
    df = make_distinct_df(pd.read_csv(csv_path))

    # τ: groups sorted DESC
    all_groups_desc = build_groups_desc(df, intent_col)

    # E: pick z groups, keep DESC order
    E_desc = choose_E(all_groups_desc, z=z, seed=seed_groups)
    zE = len(E_desc)  # should be z unless dataset has fewer

    # ranks r(e, τ(E))
    r_tau_E = rank_in_tau_E(E_desc)

    # bias levels + random b(e)
    levels = make_bias_levels(m_levels, min_bias, max_bias, include_zero=True)
    b_map = random_bias_map(E_desc, levels=levels, seed=seed_bias)

    # β(I): random top-k subset
    k = default_k(zE, frac=k_frac)
    beta_I = random_beta_I(E_desc, k=k, seed=seed_beta)

    # thresholds by bias-pair (cached by (z, levels))
    cache_key = (zE, tuple(levels))
    if cache_key not in _BOUNDS_CACHE:
        _BOUNDS_CACHE[cache_key] = compute_indifference_thresholds_by_bias_pair(zE, levels)
    bounds_by_bias = _BOUNDS_CACHE[cache_key]

    det = detect_credible_and_noncredible(E_desc, r_tau_E, b_map, bounds_by_bias, beta_I)

    return {
        "intent_col": intent_col,
        "intent_order": "DESC",
        "E_desc": E_desc,
        "z": zE,
        "k": k,
        "bias_levels": levels,
        "b_map": b_map,
        "bounds_by_bias": bounds_by_bias,
        **det,
    }


# =============================================================================
# 7) Example usage (edit paths/params)
# =============================================================================

if __name__ == "__main__":
    out = run_one(
        csv_path="../data/real/census_bucketized.csv",
        intent_col="education_num",
        z=16,              # number of groups in E
        m_levels=6,        # number of bias levels
        min_bias=0.0,
        max_bias=2.0,
        k_frac=0.50,       # β(I) returns random top 50%
        seed_groups=0,
        seed_bias=1,
        seed_beta=2,
    )

    print("Intent:", f"ORDER BY {out['intent_col']} DESC")
    print("z=|E|:", out["z"], "k:", out["k"])
    print("E (DESC):", out["E_desc"])
    print("bias levels:", out["bias_levels"])
    print("β(I) (random):", out["beta_I"])
    print("non-credible returned:", out["non_credible_returned"])
    print("credible returned:", out["credible_returned"])
    if out["non_credible_returned"]:
        e0 = out["non_credible_returned"][0]
        print("witness for", e0, ":", out["witnesses"][e0])
