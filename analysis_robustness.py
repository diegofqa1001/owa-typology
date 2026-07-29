"""
Analytical robustness study of the Fuzzy-OWA calibration chain.
Replaces the (unreproducible) synthetic expert panel with mathematical
sensitivity analyses that involve no simulated human judgement.

Outputs: JSON results + the numbers quoted in Section 4 of the manuscript.
"""
import json
import itertools
import numpy as np
from scipy.stats import kendalltau

from owa_typology import (LABELS, PROFILES, DIMENSIONS, REVERSED,
                          label_centroids, attitude_score, profile_vector,
                          rim_weights, orness, solve_alpha, owa, calibrate,
                          classify)

RNG = np.random.default_rng(2026)
OUT = {}

# ---------------------------------------------------------------- baseline
base = calibrate()
names = list(base)
OUT["baseline"] = {k: {"a": round(v["a"], 4), "alpha": round(v["alpha"], 4),
                       "orness": round(v["orness"], 6),
                       "w": [round(x, 4) for x in v["w"]]}
                   for k, v in base.items()}
print("=== Baseline calibration ===")
for k, v in base.items():
    print(f"{k:11s} a={v['a']:.4f} alpha={v['alpha']:7.4f} orness={v['orness']:.6f}")

S = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9])
F = {k: owa(S, v["w"]) for k, v in base.items()}
OUT["worked_example"] = {"S": S.tolist(), "F": {k: round(x, 4) for k, x in F.items()},
                         "spread": round(max(F.values()) - min(F.values()), 4)}
print("\nWorked example F:", {k: round(x, 3) for k, x in F.items()},
      "spread =", round(max(F.values()) - min(F.values()), 4))

# ------------------------------------------- A. label-parameterisation robustness
print("\n=== A. Robustness to label parameterisation ===")
alt_sets = {
    "baseline": LABELS,
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
lab_res = {}
base_rank = {n: i for i, n in enumerate(names)}
for tag, labs in alt_sets.items():
    cog = label_centroids(labs)
    sc = {n: attitude_score(v, cog) for n, v in PROFILES.items()}
    order = [n for n, _ in sorted(sc.items(), key=lambda kv: kv[1])]
    tau = kendalltau([base_rank[n] for n in names], [order.index(n) for n in names]).statistic
    lab_res[tag] = {"scores": {k: round(v, 4) for k, v in sc.items()},
                    "order": order, "kendall_tau_vs_baseline": round(float(tau), 4),
                    "identical_order": order == names,
                    "range": [round(min(sc.values()), 4), round(max(sc.values()), 4)]}
    print(f"{tag:24s} tau={tau:.3f} same_order={order == names} "
          f"range=[{min(sc.values()):.3f},{max(sc.values()):.3f}]")
OUT["label_robustness"] = lab_res

# --------------------------------------- B. dimension-weight robustness (Dirichlet)
print("\n=== B. Robustness to dimension weighting (10,000 Dirichlet draws) ===")
cog = label_centroids()
n_draw, same, taus, minsep = 10_000, 0, [], []
for conc in (50.0,):
    for _ in range(n_draw):
        w = RNG.dirichlet(np.full(7, conc))
        sc = {n: attitude_score(v, cog, w) for n, v in PROFILES.items()}
        order = [n for n, _ in sorted(sc.items(), key=lambda kv: kv[1])]
        same += order == names
        taus.append(kendalltau([base_rank[n] for n in names],
                               [order.index(n) for n in names]).statistic)
        vals = sorted(sc.values())
        minsep.append(min(np.diff(vals)))
OUT["dimension_weight_robustness"] = {
    "draws": n_draw, "concentration": 50.0,
    "pct_identical_order": round(100 * same / n_draw, 2),
    "mean_kendall_tau": round(float(np.mean(taus)), 4),
    "p05_kendall_tau": round(float(np.percentile(taus, 5)), 4),
    "mean_min_separation": round(float(np.mean(minsep)), 4)}
print(f"identical order in {100*same/n_draw:.2f}% of draws; "
      f"mean tau={np.mean(taus):.4f} (5th pct {np.percentile(taus,5):.4f})")

# ------------------------------ B2. breaking point under unequal weighting
print("\n=== B2. Breaking point: decreasing Dirichlet concentration ===")
bp = {}
for conc in (50, 20, 10, 5, 2, 1, 0.5, 0.2):
    same, taus, N, flips = 0, [], 5000, {}
    for _ in range(N):
        w = RNG.dirichlet(np.full(7, float(conc)))
        sc = {n: attitude_score(v, cog, w) for n, v in PROFILES.items()}
        order = [n for n, _ in sorted(sc.items(), key=lambda kv: kv[1])]
        same += order == names
        taus.append(kendalltau(range(8), [order.index(n) for n in names]).statistic)
        for i in range(7):
            if order[i] != names[i]:
                key = f"{names[i]}<->{names[i+1]}"
                flips[key] = flips.get(key, 0) + 1
                break
    bp[str(conc)] = {"pct_identical": round(100 * same / N, 1),
                     "mean_tau": round(float(np.mean(taus)), 4),
                     "first_inversion_pct": {k: round(100 * v / N, 1) for k, v in
                                             sorted(flips.items(), key=lambda kv: -kv[1])[:4]}}
    print(f"  conc={conc:5} identical {100*same/N:5.1f}%  mean tau={np.mean(taus):.4f}")
OUT["breaking_point"] = bp
OUT["adjacent_separations"] = {f"{names[i]}->{names[i+1]}":
                               round(base[names[i+1]]["a"] - base[names[i]]["a"], 4)
                               for i in range(7)}
print("  adjacent separations:", OUT["adjacent_separations"])

# ------------------------------------ C. leave-one-dimension-out (structural)
print("\n=== C. Leave-one-dimension-out ===")
loo = {}
for i in range(7):
    keep = [j for j in range(7) if j != i]
    sc = {}
    for n, vec in PROFILES.items():
        s = np.array([cog[l] for l in vec])
        t = s.copy()
        for r in REVERSED:
            t[r] = 1 - s[r]
        sc[n] = float(t[keep].mean())
    order = [n for n, _ in sorted(sc.items(), key=lambda kv: kv[1])]
    tau = kendalltau([base_rank[n] for n in names], [order.index(n) for n in names]).statistic
    loo[DIMENSIONS[i]] = {"order": order, "tau": round(float(tau), 4),
                          "identical": order == names}
    print(f"drop {DIMENSIONS[i]:30s} tau={tau:.3f} same={order == names}")
OUT["leave_one_out"] = loo

# ------------------------------------------ D. classification stability
print("\n=== D. Classification stability under measurement noise ===")
protos = {n: profile_vector(v, cog) for n, v in PROFILES.items()}
stab = {}
for sigma in (0.05, 0.10, 0.15, 0.20):
    hits, orn_err = {n: 0 for n in names}, []
    trials = 2000
    for n in names:
        p = protos[n]
        for _ in range(trials):
            r = np.clip(p + RNG.normal(0, sigma, 7), 0, 1)
            got = classify(r, PROFILES, cog)
            hits[n] += got == n
            orn_err.append(abs(base[got]["a"] - base[n]["a"]))
    stab[sigma] = {"per_profile_pct": {n: round(100 * h / trials, 1) for n, h in hits.items()},
                   "overall_pct": round(100 * sum(hits.values()) / (trials * 8), 1),
                   "mean_abs_orness_error": round(float(np.mean(orn_err)), 4)}
    print(f"sigma={sigma:.2f}  correct={stab[sigma]['overall_pct']:.1f}%  "
          f"mean |Δorness|={stab[sigma]['mean_abs_orness_error']:.4f}")
OUT["classification_stability"] = {str(k): v for k, v in stab.items()}

# ------------------------------------------ E. ranking divergence across profiles
print("\n=== E. Ranking divergence across profiles (1,000 random asset sets) ===")
taus_rank, top1 = [], []
for _ in range(1000):
    A = RNG.random((10, 7))
    rk = {n: np.argsort(-np.array([owa(row, base[n]["w"]) for row in A])) for n in names}
    taus_rank.append(kendalltau(rk["Guardian"], rk["Visionary"]).statistic)
    top1.append(rk["Guardian"][0] != rk["Visionary"][0])
OUT["ranking_divergence"] = {
    "asset_sets": 1000, "assets_per_set": 10, "criteria": 7,
    "mean_kendall_tau_guardian_vs_visionary": round(float(np.mean(taus_rank)), 4),
    "pct_different_top_asset": round(100 * float(np.mean(top1)), 1)}
print(f"mean tau(Guardian, Visionary) = {np.mean(taus_rank):.3f}; "
      f"different top asset in {100*np.mean(top1):.1f}% of sets")

# ------------------------------------------ F. comparison of orness schemes
print("\n=== F. Comparison with alternative orness schemes ===")
schemes = {
    "derived (this paper)": [base[n]["a"] for n in names],
    "equispaced 0.158-0.865": list(np.linspace(0.158, 0.865, 8)),
    "octiles (2k-1)/16": [(2 * k - 1) / 16 for k in range(1, 9)],
}
comp = {}
for tag, orns in schemes.items():
    ws = [rim_weights(solve_alpha(o)) for o in orns]
    Fv = [owa(S, w) for w in ws]
    comp[tag] = {"orness": [round(o, 4) for o in orns],
                 "F": [round(f, 4) for f in Fv],
                 "spread": round(max(Fv) - min(Fv), 4),
                 "mean_gap": round(float(np.mean(np.diff(sorted(orns)))), 4),
                 "sd_gap": round(float(np.std(np.diff(sorted(orns)))), 4)}
    print(f"{tag:24s} spread={max(Fv)-min(Fv):.3f}  "
          f"gap mean={np.mean(np.diff(sorted(orns))):.3f} sd={np.std(np.diff(sorted(orns))):.3f}")
OUT["orness_schemes"] = comp

# ------------------------------------------ G. exactness / reproducibility check
err = max(abs(v["orness"] - v["a"]) for v in base.values())
sums = [abs(v["w"].sum() - 1) for v in base.values()]
mono = all(base[names[i]]["orness"] < base[names[i + 1]]["orness"] for i in range(7))
OUT["numerical_checks"] = {"max_abs_orness_error": float(err),
                           "max_abs_weight_sum_error": float(max(sums)),
                           "strict_monotonicity": bool(mono)}
print(f"\nmax |orness - a| = {err:.2e}; max |Σw - 1| = {max(sums):.2e}; monotone = {mono}")

with open("robustness_results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nWrote robustness_results.json")
