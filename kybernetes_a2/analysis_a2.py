#!/usr/bin/env python3
"""
analysis_a2.py - Robustness of the typology construction under the ordering
rule of the thesis (risk-attitude core first, modulating dimensions second)
and the canonical octile anchors (2k - 1)/16.

Reproduces every number in Tables 2 and 3 and Section 4.3 of the manuscript
"From behavioural dimensions to an aggregation attitude" (Kybernetes submission).

Analyses
  R1  ordering rule: core and modulator means, critical core weight omega*,
      order under equal weights (the attitude score of owa_typology.py)
  R2  label parameterisation: five membership-function sets
  R3  dimension weighting: (a) within-block Dirichlet weights under the
      lexicographic rule; (b) global random weights for a convex blend
      (core weight omega ~ U(0,1)) and for a flat Dirichlet over the seven
      dimensions; first inversions
  R4  leave-one-dimension-out
  R5  classification stability (nearest prototype, Gaussian noise) and mean
      absolute orness error with octile anchors
  R6  ranking divergence Guardian vs Visionary, octile-anchor weights;
      spread of the worked example
  R7  anchor consequence of the equal-weight order (permutation within the trio)

Usage:  python kybernetes_a2/analysis_a2.py   ->  results/a2_robustness.json
Seed 2026. Python >= 3.10, numpy, scipy.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from owa_typology import rim_weights, orness, solve_alpha, owa  # noqa: E402

RNG = np.random.default_rng(2026)

# Coordinates in the canonical order of the thesis (Table 3.3):
# D1 risk tolerance, D5 loss aversion (reversed), D4 financial self-efficacy,
# D10 ambiguity tolerance, D8 investment horizon, D7 emotional regulation,
# D12 perceived social influence (reversed).
DIMS = ["D1", "D5", "D4", "D10", "D8", "D7", "D12"]
REV = (1, 6)
CORE = [0, 1]
MOD = [2, 3, 4, 5, 6]

PROFILES = {  # rank order of the thesis
    "Guardian":   ["VL", "VH", "VL", "VL", "VL", "VL", "VH"],
    "Sentinel":   ["L",  "H",  "M",  "L",  "M",  "L",  "H"],
    "Pragmatist": ["M",  "M",  "M",  "M",  "M",  "M",  "M"],
    "Analyst":    ["M",  "M",  "VH", "H",  "H",  "H",  "VL"],
    "Strategist": ["H",  "M",  "H",  "M",  "H",  "H",  "L"],
    "Adventurer": ["H",  "L",  "H",  "M",  "H",  "M",  "H"],
    "Innovator":  ["VH", "L",  "VH", "H",  "VH", "H",  "L"],
    "Visionary":  ["VH", "VL", "VH", "VH", "VH", "VH", "VL"],
}
NAMES = list(PROFILES)
ANCHOR = {n: (2 * k - 1) / 16 for k, n in enumerate(NAMES, start=1)}

LABEL_SETS = {
    "baseline": {
        "VL": (0.00, 0.00, 0.10, 0.25), "L": (0.10, 0.25, 0.25, 0.40),
        "M": (0.30, 0.45, 0.55, 0.70), "H": (0.60, 0.75, 0.75, 0.90),
        "VH": (0.75, 0.90, 1.00, 1.00)},
    "equidistant_triangular": {
        "VL": (0.0, 0.0, 0.0, 0.25), "L": (0.0, 0.25, 0.25, 0.5),
        "M": (0.25, 0.5, 0.5, 0.75), "H": (0.5, 0.75, 0.75, 1.0),
        "VH": (0.75, 1.0, 1.0, 1.0)},
    "wide_overlap": {
        "VL": (0.0, 0.0, 0.15, 0.35), "L": (0.05, 0.25, 0.30, 0.50),
        "M": (0.25, 0.45, 0.55, 0.75), "H": (0.50, 0.70, 0.75, 0.95),
        "VH": (0.65, 0.85, 1.0, 1.0)},
    "narrow_overlap": {
        "VL": (0.0, 0.0, 0.05, 0.15), "L": (0.10, 0.20, 0.28, 0.35),
        "M": (0.35, 0.47, 0.53, 0.65), "H": (0.65, 0.72, 0.80, 0.90),
        "VH": (0.85, 0.95, 1.0, 1.0)},
    "skewed_conservative": {
        "VL": (0.0, 0.0, 0.12, 0.30), "L": (0.08, 0.22, 0.30, 0.45),
        "M": (0.32, 0.44, 0.52, 0.65), "H": (0.58, 0.70, 0.76, 0.88),
        "VH": (0.80, 0.92, 1.0, 1.0)},
}


def cog(a, b, c, d):
    """Closed-form centre of gravity of a trapezoid (a, b, c, d)."""
    return ((d * d + c * c + c * d) - (a * a + b * b + a * b)) / (3.0 * (d + c - a - b))


def cogs(labels):
    return {k: cog(*v) for k, v in labels.items()}


def raw(vec, cg):
    return np.array([cg[l] for l in vec])


def directional(vec, cg):
    t = raw(vec, cg)
    for i in REV:
        t[i] = 1.0 - t[i]
    return t


def lex_order(cg, core_w=None, mod_w=None, keep=None):
    """Lexicographic rule: core mean first, modulator mean as tie-breaker."""
    core = CORE if keep is None else [i for i in CORE if i in keep]
    mod = MOD if keep is None else [i for i in MOD if i in keep]
    cw = np.full(len(core), 1 / len(core)) if core_w is None else np.asarray(core_w)
    mw = np.full(len(mod), 1 / len(mod)) if mod_w is None else np.asarray(mod_w)
    key = {}
    for n, v in PROFILES.items():
        t = directional(v, cg)
        key[n] = (round(float(cw @ t[core]), 12) if core else 0.0,
                  round(float(mw @ t[mod]), 12) if mod else 0.0)
    return sorted(NAMES, key=lambda n: key[n])


def weighted_order(cg, w):
    sc = {n: float(np.asarray(w) @ directional(v, cg)) for n, v in PROFILES.items()}
    return sorted(NAMES, key=sc.get)


def tau(order):
    return float(kendalltau(range(8), [order.index(n) for n in NAMES]).statistic)


def first_inversion(order):
    """First adjacent pair of the thesis order that appears reversed."""
    pos = {n: i for i, n in enumerate(order)}
    for i in range(7):
        a, b = NAMES[i], NAMES[i + 1]
        if pos[a] > pos[b]:
            return f"{a}>{b}"
    return None


def main():
    out = {}
    base = cogs(LABEL_SETS["baseline"])
    out["cog_labels"] = {k: round(v, 4) for k, v in base.items()}

    # --------------------------------------------------------------- R1
    prof = {}
    for k, n in enumerate(NAMES, start=1):
        t = directional(PROFILES[n], base)
        a = ANCHOR[n]
        prof[n] = {"vector": PROFILES[n], "rank_k": k,
                   "core": round(float(t[CORE].mean()), 4),
                   "modulators": round(float(t[MOD].mean()), 4),
                   "equal_weight_score": round(float(t.mean()), 4),
                   "octile_anchor": a, "rim_beta": round(solve_alpha(a), 4),
                   "weights": [round(x, 4) for x in rim_weights(solve_alpha(a))]}
    out["profiles"] = prof
    lex = lex_order(base)
    eq = weighted_order(base, np.full(7, 1 / 7))

    def thr(p, q):  # core weight above which p < q in a convex blend
        tp, tq = directional(PROFILES[p], base), directional(PROFILES[q], base)
        dn = tq[CORE].mean() - tp[CORE].mean()
        dm = tp[MOD].mean() - tq[MOD].mean()
        return dm / (dn + dm) if dm > 0 else 0.0

    grid = np.linspace(0, 1, 20001)
    ok = [w for w in grid if weighted_order(
        base, np.r_[np.full(2, w / 2), np.full(5, (1 - w) / 5)]) == NAMES]
    out["R1_ordering_rule"] = {
        "lexicographic_order": lex, "reproduces_thesis": lex == NAMES,
        "equal_weight_order": eq, "equal_weight_core_share": round(2 / 7, 4),
        "kendall_tau_equal_vs_thesis": round(tau(eq), 4),
        "critical_core_weight_Analyst_Strategist": round(thr("Analyst", "Strategist"), 4),
        "critical_core_weight_Strategist_Adventurer": round(thr("Strategist", "Adventurer"), 4),
        "critical_core_weight_Pragmatist_Analyst": round(thr("Pragmatist", "Analyst"), 4),
        "core_weight_interval_reproducing_thesis": [round(min(ok), 4), round(max(ok), 4)]}

    # --------------------------------------------------------------- R2
    r2 = {}
    for tag, labs in LABEL_SETS.items():
        cg = cogs(labs)
        o = lex_order(cg)
        e = weighted_order(cg, np.full(7, 1 / 7))
        r2[tag] = {"lexicographic_identical": o == NAMES, "tau_lex": round(tau(o), 4),
                   "equal_weight_order": e, "tau_equal": round(tau(e), 4)}
    out["R2_label_parameterisation"] = r2

    # --------------------------------------------------------------- R3
    n_draw = 10_000
    same = 0
    for _ in range(n_draw):
        o = lex_order(base, RNG.dirichlet(np.ones(2)), RNG.dirichlet(np.ones(5)))
        same += o == NAMES
    r3a = {"draws": n_draw, "pct_identical": round(100 * same / n_draw, 2)}

    def global_draw(kind):
        if kind == "hierarchical":
            om = RNG.uniform()
            return np.r_[om * RNG.dirichlet(np.ones(2)), (1 - om) * RNG.dirichlet(np.ones(5))]
        return RNG.dirichlet(np.ones(7))

    r3b = {}
    for kind in ("hierarchical", "flat_dirichlet"):
        same, taus, flips, core_share, outer, outer5, above, above_same = 0, [], {}, [], 0, 0, 0, 0
        for _ in range(n_draw):
            w = global_draw(kind)
            o = weighted_order(base, w)
            same += o == NAMES
            outer += all(o[i] == NAMES[i] for i in (0, 1, 6, 7))
            outer5 += all(o[i] == NAMES[i] for i in (0, 1, 2, 6, 7))
            if w[CORE].sum() > 6 / 11:
                above += 1
                above_same += o == NAMES
            taus.append(tau(o))
            core_share.append(w[CORE].sum())
            f = first_inversion(o)
            if f:
                flips[f] = flips.get(f, 0) + 1
        nflip = n_draw - same
        r3b[kind] = {"pct_identical": round(100 * same / n_draw, 2),
                     "mean_tau": round(float(np.mean(taus)), 4),
                     "p05_tau": round(float(np.percentile(taus, 5)), 4),
                     "mean_core_share": round(float(np.mean(core_share)), 4),
                     "pct_extremes_unchanged": round(100 * outer / n_draw, 2),
                     "pct_only_trio_reordered_or_identical": round(100 * outer5 / n_draw, 2),
                     "pct_identical_given_core_share_above_6_11":
                         round(100 * above_same / above, 2) if above else None,
                     "draws_core_share_above_6_11": above,
                     "first_inversions_pct_of_changed": {
                         k: round(100 * v / nflip, 1)
                         for k, v in sorted(flips.items(), key=lambda kv: -kv[1])[:4]}}
    out["R3_dimension_weighting"] = {"within_block_lexicographic": r3a, "global": r3b}

    # --------------------------------------------------------------- R4
    r4 = {}
    for i, d in enumerate(DIMS):
        keep = [j for j in range(7) if j != i]
        o = lex_order(base, keep=keep)
        r4[d] = {"identical": o == NAMES, "tau": round(tau(o), 4),
                 "first_inversion": first_inversion(o)}
    out["R4_leave_one_out"] = r4

    # --------------------------------------------------------------- R5
    protos = {n: raw(v, base) for n, v in PROFILES.items()}
    P = np.array([protos[n] for n in NAMES])
    r5 = {}
    for sigma in (0.05, 0.10, 0.15, 0.20):
        hits, err, adj = 0, [], 0
        trials = 5000
        for k, n in enumerate(NAMES):
            X = np.clip(protos[n] + RNG.normal(0, sigma, (trials, 7)), 0, 1)  # clipped to [0, 1]
            got = np.argmin(((X[:, None, :] - P[None, :, :]) ** 2).sum(-1), axis=1)
            hits += int((got == k).sum())
            adj += int((np.abs(got - k) <= 1).sum())
            err.extend(np.abs(np.array([ANCHOR[NAMES[g]] for g in got]) - ANCHOR[n]))
        r5[str(sigma)] = {"pct_correct": round(100 * hits / (8 * trials), 1),
                          "pct_within_one_rank": round(100 * adj / (8 * trials), 1),
                          "mean_abs_orness_error": round(float(np.mean(err)), 4)}
    out["R5_classification_stability"] = r5

    # --------------------------------------------------------------- R6
    W = {n: rim_weights(solve_alpha(ANCHOR[n])) for n in NAMES}
    taus_r, top, top_pa, taus_pa = [], [], [], []
    for _ in range(1000):
        A = RNG.random((10, 7))  # ten assets, seven criterion scores ~ U(0, 1)
        sc = {n: np.array([owa(r, W[n]) for r in A])
              for n in ("Guardian", "Visionary", "Pragmatist", "Analyst")}
        taus_r.append(kendalltau(sc["Guardian"], sc["Visionary"]).statistic)
        top.append(np.argmax(sc["Guardian"]) != np.argmax(sc["Visionary"]))
        taus_pa.append(kendalltau(sc["Pragmatist"], sc["Analyst"]).statistic)
        top_pa.append(np.argmax(sc["Pragmatist"]) != np.argmax(sc["Analyst"]))
    S = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9])
    F = {n: round(owa(S, W[n]), 4) for n in NAMES}
    out["R6_ranking_divergence"] = {
        "mean_tau_guardian_visionary": round(float(np.mean(taus_r)), 4),
        "pct_different_top_asset": round(100 * float(np.mean(top)), 1),
        "mean_tau_pragmatist_analyst": round(float(np.mean(taus_pa)), 4),
        "pct_different_top_asset_pragmatist_analyst": round(100 * float(np.mean(top_pa)), 1),
        "worked_example": {"S": S.tolist(), "F": F,
                           "spread": round(max(F.values()) - min(F.values()), 4),
                           "monotone_in_rank": all(F[NAMES[i]] < F[NAMES[i + 1]] for i in range(7))}}

    # --------------------------------------------------------------- R7
    alt = {n: (2 * k - 1) / 16 for k, n in enumerate(eq, start=1)}
    shift = {n: round(alt[n] - ANCHOR[n], 4) for n in NAMES}
    out["R7_anchor_consequence_equal_weights"] = {
        "anchors_equal_weight_order": alt, "shift": shift,
        "profiles_affected": [n for n in NAMES if shift[n] != 0],
        "max_abs_shift": max(abs(v) for v in shift.values())}

    # numerical checks
    out["checks"] = {
        "max_abs_orness_error": max(abs(orness(W[n]) - ANCHOR[n]) for n in NAMES),
        "max_abs_weight_sum_error": max(abs(W[n].sum() - 1) for n in NAMES)}

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "a2_robustness.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
