"""Test suite for owa_typology: mathematical properties and published numbers."""
import numpy as np
import pytest

from owa_typology import (LABELS, PROFILES, REVERSED, label_centroids, membership,
                          centroid, attitude_score, profile_vector, rim_weights,
                          orness, solve_alpha, owa, calibrate, classify, recommend)

COG = label_centroids()
CAL = calibrate()
NAMES = list(CAL)
S_EXAMPLE = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9])


# ----------------------------------------------------------------- fuzzy layer
def test_membership_core_and_support():
    assert membership(0.5, "M") == pytest.approx(1.0)
    assert membership(0.0, "VL") == pytest.approx(1.0)      # left shoulder
    assert membership(1.0, "VH") == pytest.approx(1.0)      # right shoulder
    assert membership(0.95, "VL") == 0.0
    assert membership(0.0, "VH") == 0.0


def test_membership_monotone_on_flanks():
    xs = np.linspace(0.30, 0.45, 20)
    vals = [membership(float(x), "M") for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_label_centroids_ordered_and_symmetric():
    c = [COG[k] for k in ("VL", "L", "M", "H", "VH")]
    assert all(b > a for a, b in zip(c, c[1:]))
    assert COG["M"] == pytest.approx(0.5, abs=1e-6)
    assert COG["VL"] + COG["VH"] == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------- attitude score
def test_neutral_profile_scores_one_half():
    assert attitude_score(PROFILES["Pragmatist"], COG) == pytest.approx(0.5, abs=1e-9)


def test_attitude_score_respects_reversal():
    """Raising loss aversion (D2) must LOWER the attitude score."""
    base = ["M"] * 7
    hi = base.copy(); hi[1] = "VH"
    lo = base.copy(); lo[1] = "VL"
    assert attitude_score(hi, COG) < attitude_score(base, COG) < attitude_score(lo, COG)


def test_attitude_score_direct_dimension_increases():
    base = ["M"] * 7
    hi = base.copy(); hi[0] = "VH"
    assert attitude_score(hi, COG) > attitude_score(base, COG)


def test_attitude_scores_in_unit_interval():
    assert all(0.0 < v["a"] < 1.0 for v in CAL.values())


# ---------------------------------------------------------------- OWA / RIM
def test_weights_sum_to_one():
    for a in (0.1, 0.5, 1.0, 2.0, 6.1):
        assert rim_weights(a).sum() == pytest.approx(1.0, abs=1e-12)


def test_orness_of_uniform_weights_is_one_half():
    assert orness(np.full(7, 1 / 7)) == pytest.approx(0.5, abs=1e-12)


def test_orness_limits():
    assert orness(rim_weights(0.001)) > 0.99      # OR-like
    assert orness(rim_weights(500.0)) < 0.01      # AND-like


def test_orness_strictly_decreasing_in_alpha():
    alphas = np.linspace(0.05, 8.0, 60)
    o = [orness(rim_weights(a)) for a in alphas]
    assert all(b < a for a, b in zip(o, o[1:]))


def test_solve_alpha_is_exact():
    for target in (0.05, 0.25, 0.5, 0.73, 0.95):
        assert orness(rim_weights(solve_alpha(target))) == pytest.approx(target, abs=1e-9)


def test_solve_alpha_neutral_is_exactly_one():
    assert solve_alpha(0.5) == pytest.approx(1.0, abs=1e-12)


def test_solve_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        solve_alpha(0.0)
    with pytest.raises(ValueError):
        solve_alpha(1.0)


def test_owa_bounds_and_special_cases():
    v = [0.2, 0.9, 0.5]
    assert min(v) <= owa(v, np.full(3, 1 / 3)) <= max(v)
    assert owa(v, [1, 0, 0]) == pytest.approx(max(v))       # pure OR
    assert owa(v, [0, 0, 1]) == pytest.approx(min(v))       # pure AND


def test_owa_is_symmetric_in_input_order():
    v = [0.1, 0.4, 0.7, 0.2, 0.9, 0.3, 0.6]
    w = rim_weights(0.7)
    assert owa(v, w) == pytest.approx(owa(list(reversed(v)), w))


# ------------------------------------------------------------- calibration
def test_calibration_is_exact_for_every_profile():
    for name, r in CAL.items():
        assert r["orness"] == pytest.approx(r["a"], abs=1e-9), name
        assert r["w"].sum() == pytest.approx(1.0, abs=1e-12), name


def test_calibration_preserves_ordering():
    a = [CAL[n]["a"] for n in NAMES]
    o = [CAL[n]["orness"] for n in NAMES]
    assert all(x < y for x, y in zip(a, a[1:]))
    assert all(x < y for x, y in zip(o, o[1:]))


def test_published_profile_order():
    assert NAMES == ["Guardian", "Sentinel", "Pragmatist", "Adventurer",
                     "Strategist", "Analyst", "Innovator", "Visionary"]


def test_published_orness_values():
    expected = {"Guardian": 0.093, "Sentinel": 0.321, "Pragmatist": 0.500,
                "Adventurer": 0.607, "Strategist": 0.679, "Analyst": 0.723,
                "Innovator": 0.817, "Visionary": 0.907}
    for name, val in expected.items():
        assert CAL[name]["orness"] == pytest.approx(val, abs=5e-4), name


def test_published_worked_example():
    expected = {"Guardian": 0.356, "Sentinel": 0.493, "Pragmatist": 0.600,
                "Adventurer": 0.664, "Strategist": 0.707, "Analyst": 0.734,
                "Innovator": 0.790, "Visionary": 0.844}
    for name, val in expected.items():
        assert owa(S_EXAMPLE, CAL[name]["w"]) == pytest.approx(val, abs=1e-3), name


def test_published_spread():
    F = [owa(S_EXAMPLE, r["w"]) for r in CAL.values()]
    assert max(F) - min(F) == pytest.approx(0.4886, abs=1e-3)


def test_pragmatist_recovers_arithmetic_mean():
    w = CAL["Pragmatist"]["w"]
    assert np.allclose(w, 1 / 7, atol=1e-9)
    assert owa(S_EXAMPLE, w) == pytest.approx(S_EXAMPLE.mean(), abs=1e-9)


def test_ordering_invariant_to_label_parameterisation():
    alt = {"VL": (0.0, 0.0, 0.0, 0.25), "L": (0.0, 0.25, 0.25, 0.5),
           "M": (0.25, 0.5, 0.5, 0.75), "H": (0.5, 0.75, 0.75, 1.0),
           "VH": (0.75, 1.0, 1.0, 1.0)}
    cog2 = label_centroids(alt)
    order = [n for n, _ in sorted(((n, attitude_score(v, cog2))
                                   for n, v in PROFILES.items()), key=lambda kv: kv[1])]
    assert order == NAMES


def test_ordering_invariant_to_leaving_out_any_dimension():
    for i in range(7):
        keep = [j for j in range(7) if j != i]
        sc = {}
        for n, vec in PROFILES.items():
            s = np.array([COG[l] for l in vec]); t = s.copy()
            for r in REVERSED:
                t[r] = 1 - s[r]
            sc[n] = t[keep].mean()
        assert [n for n, _ in sorted(sc.items(), key=lambda kv: kv[1])] == NAMES, i


# --------------------------------------------------------- classification
def test_prototypes_classify_to_themselves():
    for name, vec in PROFILES.items():
        assert classify(profile_vector(vec, COG), PROFILES, COG) == name


def test_classification_stable_under_small_noise():
    rng = np.random.default_rng(2026)
    ok = 0
    for name, vec in PROFILES.items():
        p = profile_vector(vec, COG)
        for _ in range(200):
            r = np.clip(p + rng.normal(0, 0.05, 7), 0, 1)
            ok += classify(r, PROFILES, COG) == name
    assert ok / (200 * 8) > 0.98


# ------------------------------------------------------------ recommendation
def test_recommend_returns_consistent_ranking():
    responses = profile_vector(PROFILES["Visionary"], COG)
    assets = np.array([[0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9],
                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    name, a, w, evals, ranking = recommend(responses, assets, PROFILES, COG)
    assert name == "Visionary"
    assert evals[ranking[0]] >= evals[ranking[-1]]
    assert w.sum() == pytest.approx(1.0)


def test_recommend_supports_criteria_count_other_than_seven():
    responses = profile_vector(PROFILES["Guardian"], COG)
    assets = np.array([[0.9, 0.2, 0.5, 0.4]])       # m = 4 criteria
    name, a, w, evals, _ = recommend(responses, assets, PROFILES, COG)
    assert name == "Guardian"
    assert len(w) == 4
    assert orness(w) == pytest.approx(a, abs=1e-6)


def test_conservative_profile_scores_dispersed_asset_below_neutral():
    disp = [0.95, 0.05, 0.9, 0.1, 0.85, 0.15, 0.5]
    assert owa(disp, CAL["Guardian"]["w"]) < owa(disp, CAL["Pragmatist"]["w"]) \
        < owa(disp, CAL["Visionary"]["w"])
