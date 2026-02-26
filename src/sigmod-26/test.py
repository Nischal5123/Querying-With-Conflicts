# from experiment_runner import PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc, algorithm_1_credibility_detection, run_pipeline_domain, make_random_multilevel_bias

from coi_algorithms import (
    PriorSpec, Bias1D, biases_from_bias_obj, dedupe_and_sort_desc,
    compute_expected_posteriors, system_best_response, algorithm_1_credibility_detection,
    algorithm_2_build_qbase, algorithm_4_maximally_informative,
    user_utility_from_response, run_pipeline_domain, make_random_multilevel_bias
)
# =============================================================================
# Tests (focus on threshold)
# =============================================================================


def _pairs_count(n: int) -> int:
    return n * (n - 1) // 2

def run_tests():
    print("\n=== TESTS (threshold only) ===")
    prior = PriorSpec(kind="uniform")

    # Test 1: symmetry (zero bias) → all pairs credible, q_base==singletons, q_star==q_base
    dom = [5, 4, 3, 2, 1]
    zero_bias = Bias1D(kind="constant", base=0.0)
    biases = biases_from_bias_obj(dedupe_and_sort_desc(dom), zero_bias)

    # Alg.1 credible set size = all strict pairs
    q0 = [[x] for x in dedupe_and_sort_desc(dom)]
    C = algorithm_1_credibility_detection(q0, dedupe_and_sort_desc(dom), prior, biases=biases)
    assert len(C) == _pairs_count(len(dom)), f"Symmetry: expected all pairs credible, got |C|={len(C)}"

    out = run_pipeline_domain(dom, prior, zero_bias, receiver_model="threshold")
    q_base, q_star = out["q_base"], out["q_star"]
    assert all(len(g) == 1 for g in q_base), "Symmetry: q_base should keep singletons"
    assert q_star == q_base, "Symmetry: q_star should equal q_base (no positive gains)"
    print("Test 1 passed ✓ (symmetry)")

    # Test 2: window bias forces at least one merge
    dom2 = [10, 9, 8, 7, 6, 5]
    win_bias = Bias1D(kind="window", lo=7.0, hi=8.0, height=0.9)  # favors mid band
    out2 = run_pipeline_domain(dom2, prior, win_bias, receiver_model="threshold")
    q_base2 = out2["q_base"]
    print("q_base2:", q_base2)
    assert any(len(g) > 1 for g in q_base2), "Window bias: expected at least one merge in q_base"
    print("Test 2 passed ✓ (window bias causes merge)")

    # Test 3: multilevel random bias runs and produces a valid partition
    dom3 = [12, 11, 10, 9, 8, 7, 6]
    ml_bias = make_random_multilevel_bias(dom3, levels=(0.85, 0.55, 0.25, 0.0), probs=(0.25, 0.25, 0.25, 0.25), seed=7)
    out3 = run_pipeline_domain(dom3, prior, ml_bias, receiver_model="threshold")
    q_base3, q_star3 = out3["q_base"], out3["q_star"]
    assert len(q_base3) >= 1 and sum(len(g) for g in q_base3) == len(set(dom3)), "Multilevel: q_base invalid"
    assert len(q_star3) >= 1 and sum(len(g) for g in q_star3) == len(set(dom3)), "Multilevel: q_star invalid"
    print("Test 3 passed ✓ (multilevel bias pipeline)")

    # Test 4: no-positive-gain safeguard (construct case with negative/zero gains)
    # Using symmetry again ensures all merges have ≤0 marginal gain due to order bonus normalization.
    out4 = run_pipeline_domain(dom, prior, zero_bias, receiver_model="threshold")
    assert out4["q_star"] == out4["q_base"], "No-positive-gain safeguard failed: expected q_star == q_base"
    print("Test 4 passed ✓ (no-positive-gain safeguard)")

    print("All tests passed ✅")


if __name__ == "__main__":
    run_tests()