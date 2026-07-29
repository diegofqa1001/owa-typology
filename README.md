# owa-typology

Reference implementation and reproducibility package for:

> **From Behavioral Finance to Explainable Investment Recommendation: A Fuzzy-OWA Typology of Investor Risk Profiles**
> Diego Quintero-Avellaneda (Universidad Nacional de Colombia, Sede Manizales)

This repository reproduces **every number, table and figure** in the paper.

## What the framework does

It maps a behavioral characterization of an investor onto an aggregation attitude, with no free parameter in between:

```
linguistic vector  →  attitude score  →  orness  →  RIM quantifier  →  OWA weights  →  asset evaluation
```

Each profile is a seven-component fuzzy linguistic vector over the behavioral dimensions D1–D7. Its attitude score is computed by defuzzifying the labels (centre of gravity) and averaging, with the direction of D2 (loss aversion) and D7 (perceived social influence) reversed. That score becomes the target orness, which uniquely determines the exponent α of Yager's RIM quantifier Q(r) = r^α and hence the OWA weight vector. Nothing is assigned by hand after the typology is stated.

## Install and run

```bash
python -m pip install -r requirements.txt
python analysis_robustness.py     # all analyses -> robustness_results.json
python make_figures.py            # Figures 1-3 (300 dpi, Okabe-Ito, white background)
python -m pytest -q               # test suite
```

Python ≥ 3.10. Dependencies: numpy, scipy, matplotlib, pytest.

## Files

| File | Purpose |
|---|---|
| `owa_typology.py` | Reference implementation: labels, profiles, attitude score, RIM/OWA calibration, classification, recommendation |
| `analysis_robustness.py` | Section 4 analyses: exactness, label-parameterization and dimension-weight robustness, breaking-point study, leave-one-out, classification stability, ranking divergence, comparison of orness schemes |
| `make_figures.py` | Figures 1–3 |
| `test_owa_typology.py` | Test suite covering the mathematical properties and the published numbers |
| `robustness_results.json` | Machine-readable output of all analyses |
| `instrument/` | Content-validation instrument and Delphi protocol prepared for the confirmatory phase |

## Reproducing the paper's numbers

```python
from owa_typology import calibrate, owa
import numpy as np

cal = calibrate()                       # Tables 3 and 4
for name, r in cal.items():
    print(f"{name:11s} a={r['a']:.4f} alpha={r['alpha']:7.4f} orness={r['orness']:.6f}")

S = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9])   # worked example, Section 3.3
F = {n: owa(S, r['w']) for n, r in cal.items()}
print("spread =", round(max(F.values()) - min(F.values()), 4))   # 0.4886
```

Using the framework on a new investor:

```python
from owa_typology import recommend
responses = [0.72, 0.30, 0.81, 0.65, 0.78, 0.70, 0.20]   # D1..D7 in [0,1]
assets = [[0.8,0.6,0.7,0.5,0.4,0.3,0.9], [0.5,0.5,0.6,0.7,0.6,0.5,0.4]]
profile, attitude, w, evals, ranking = recommend(responses, assets)
print(profile, round(attitude, 3), evals.round(3), ranking)
```

## Scope and limitations

The typology is a **conceptually derived** classification (a typology, not a taxonomy): it is constructed from a synthesis of the empirical behavioral finance literature and has **not** been derived from, or validated against, investor data. The analyses in `analysis_robustness.py` establish internal properties of the formal construction — exactness, invariance, stability, discriminative power — not empirical validity. The content-validation study with a human expert panel and the empirical study with real investors are the next stages of the research program; the instrument for the former is included here.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
