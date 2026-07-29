"""
owa_typology — Fuzzy-OWA typology of investor risk profiles.

Reference implementation of the calibration chain:
    linguistic vector -> attitude score -> orness -> RIM quantifier -> OWA weights

The chain contains NO free parameter once the typology and two stated
conventions (label parameterisation, direction-reversal set) are fixed.

Author: Diego Quintero-Avellaneda
License: MIT
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Linguistic labels: trapezoidal fuzzy numbers (a, b, c, d) on [0, 1]
# --------------------------------------------------------------------------
LABELS = {
    "VL": (0.00, 0.00, 0.10, 0.25),
    "L":  (0.10, 0.25, 0.25, 0.40),
    "M":  (0.30, 0.45, 0.55, 0.70),
    "H":  (0.60, 0.75, 0.75, 0.90),
    "VH": (0.75, 0.90, 1.00, 1.00),
}

# Dimensions D1..D7 (see Table 2 of the manuscript)
DIMENSIONS = [
    "D1 Risk tolerance",
    "D2 Loss aversion",
    "D3 Financial self-efficacy",
    "D4 Ambiguity tolerance",
    "D5 Investment horizon",
    "D6 Emotional regulation",
    "D7 Perceived social influence",
]

# Dimensions whose direction is reversed w.r.t. decision optimism (0-indexed)
REVERSED = (1, 6)  # D2 loss aversion, D7 perceived social influence

# Prototypical profiles: raw linguistic labels in D1..D7 order
PROFILES = {
    "Guardian":   ["VL", "VH", "VL", "VL", "VL", "VL", "VH"],
    "Sentinel":   ["L",  "H",  "M",  "L",  "M",  "L",  "H"],
    "Pragmatist": ["M",  "M",  "M",  "M",  "M",  "M",  "M"],
    "Adventurer": ["H",  "L",  "H",  "M",  "H",  "M",  "H"],
    "Strategist": ["H",  "M",  "H",  "M",  "H",  "H",  "L"],
    "Analyst":    ["M",  "M",  "VH", "H",  "H",  "H",  "VL"],
    "Innovator":  ["VH", "L",  "VH", "H",  "VH", "H",  "L"],
    "Visionary":  ["VH", "VL", "VH", "VH", "VH", "VH", "VL"],
}


# --------------------------------------------------------------------------
# Fuzzy layer
# --------------------------------------------------------------------------
def membership(x: float, label: str) -> float:
    """Trapezoidal membership degree of x in `label` (shoulders supported)."""
    a, b, c, d = LABELS[label]
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return 1.0 if b == a else (x - a) / (b - a)
    return 1.0 if d == c else (d - x) / (d - c)


def centroid(label: str, grid: int = 100_001) -> float:
    """Centre-of-gravity defuzzification of a linguistic label."""
    a, b, c, d = LABELS[label]
    xs = np.linspace(0.0, 1.0, grid)
    mu = np.array([membership(float(x), label) for x in xs])
    return float((xs * mu).sum() / mu.sum())


def label_centroids(labels: dict | None = None) -> dict:
    """COG value of every label (cached-friendly helper)."""
    global LABELS
    if labels is not None:
        old, LABELS = LABELS, labels
        try:
            return {k: centroid(k) for k in LABELS}
        finally:
            LABELS = old
    return {k: centroid(k) for k in LABELS}


def attitude_score(vector, cog: dict | None = None, weights=None) -> float:
    """
    Attitude score of a profile, derived from its linguistic vector.

    ā = Σ_i ω_i · t_i,  where t_i = s_i for direct dimensions and
    t_i = 1 − s_i for direction-reversed ones (D2, D7), s_i the COG of the
    label, and ω the dimension weights (uniform by default).
    """
    cog = cog or label_centroids()
    s = np.array([cog[lab] for lab in vector], dtype=float)
    t = s.copy()
    for i in REVERSED:
        t[i] = 1.0 - s[i]
    w = np.full(len(t), 1.0 / len(t)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return float((w * t).sum())


def profile_vector(vector, cog: dict | None = None) -> np.ndarray:
    """Defuzzified RAW label vector of a profile (classification space)."""
    cog = cog or label_centroids()
    return np.array([cog[lab] for lab in vector], dtype=float)


# --------------------------------------------------------------------------
# OWA / RIM layer
# --------------------------------------------------------------------------
def rim_weights(alpha: float, n: int = 7) -> np.ndarray:
    """OWA weights from the RIM quantifier Q(r) = r^alpha."""
    j = np.arange(1, n + 1)
    return (j / n) ** alpha - ((j - 1) / n) ** alpha


def orness(w) -> float:
    """Yager's orness (attitudinal character) of an OWA weight vector."""
    w = np.asarray(w, float)
    n = len(w)
    j = np.arange(1, n + 1)
    return float(((n - j) * w).sum() / (n - 1))


def solve_alpha(target: float, n: int = 7, tol: float = 1e-12) -> float:
    """alpha such that orness(rim_weights(alpha, n)) == target (unique)."""
    if not 0.0 < target < 1.0:
        raise ValueError("target orness must lie in (0, 1)")
    if abs(target - 0.5) < 1e-15:
        return 1.0  # exact analytic solution for any n
    return float(brentq(lambda a: orness(rim_weights(a, n)) - target,
                        1e-9, 500.0, xtol=tol))


def owa(values, w) -> float:
    """OWA aggregation of `values` with weight vector `w`."""
    b = np.sort(np.asarray(values, float))[::-1]
    return float((np.asarray(w, float) * b).sum())


# --------------------------------------------------------------------------
# End-to-end calibration
# --------------------------------------------------------------------------
def calibrate(profiles: dict | None = None, cog: dict | None = None,
              dim_weights=None, n: int = 7) -> dict:
    """
    Full calibration chain for every profile.

    Returns {name: {'a': attitude, 'alpha': alpha, 'w': weights,
                    'orness': realised orness}}, ordered by attitude score.
    """
    profiles = profiles or PROFILES
    cog = cog or label_centroids()
    out = {}
    for name, vec in profiles.items():
        a = attitude_score(vec, cog, dim_weights)
        al = solve_alpha(a, n)
        w = rim_weights(al, n)
        out[name] = {"a": a, "alpha": al, "w": w, "orness": orness(w)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["a"]))


def classify(responses, profiles: dict | None = None, cog: dict | None = None) -> str:
    """Nearest prototypical profile (Euclidean, raw response space)."""
    profiles = profiles or PROFILES
    cog = cog or label_centroids()
    r = np.asarray(responses, float)
    d = {name: float(np.linalg.norm(r - profile_vector(vec, cog)))
         for name, vec in profiles.items()}
    return min(d, key=d.get)


def recommend(responses, asset_scores, profiles: dict | None = None,
              cog: dict | None = None):
    """
    Profile-conditioned evaluation of assets.

    `asset_scores`: array (n_assets, m) of normalised criterion scores;
    m need not equal the number of behavioural dimensions.
    Returns (profile, orness, weights, evaluations, ranking).
    """
    profiles = profiles or PROFILES
    cog = cog or label_centroids()
    name = classify(responses, profiles, cog)
    a = attitude_score(profiles[name], cog)
    A = np.atleast_2d(np.asarray(asset_scores, float))
    m = A.shape[1]
    w = rim_weights(solve_alpha(a, m), m)
    evals = np.array([owa(row, w) for row in A])
    return name, a, w, evals, np.argsort(-evals)
