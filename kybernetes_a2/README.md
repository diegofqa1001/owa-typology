# kybernetes_a2/ — reproducibility for the typology construction paper

Scripts behind Tables 2–3 and Figure 1 of the manuscript *From behavioural dimensions to an
aggregation attitude: a literature-grounded, expert-validated fuzzy-OWA typology of investor
risk profiles* (submitted to *Kybernetes*).

```bash
python kybernetes_a2/analysis_a2.py      # -> results/a2_robustness.json (seed 2026, ~20 s)
python kybernetes_a2/make_figure_a2.py   # -> kybernetes_a2/figure1_a2.png
```

What the analysis uses:

- **Ordering rule** (thesis Chapter 3): profiles are ranked lexicographically. The first key is
  the risk-attitude core, the mean of D1 and 1 − D5. Ties are broken by the mean of the modulating
  dimensions D4, D10, D8, D7 and 1 − D12.
- **Orness anchors:** α_k = (2k − 1)/16 by rank k (octile anchors). Each anchor sets the exponent
  β_k of the RIM quantifier Q(r) = r^β for n = 7, found by Brent's method with xtol = 1e-12.
- **Equal-weight attitude score** (`owa_typology.attitude_score`): reported only as a sensitivity
  case. It reorders the Analyst, Strategist and Adventurer.

The analyses are labelled R1–R7 in the docstring of `analysis_a2.py`:

- ordering rule and critical core weight (6/11);
- label parameterisation;
- within-block and global random weights;
- leave one dimension out;
- classification under noise;
- ranking divergence;
- anchor consequence of the equal-weight order.

The figures in Table 1 (PRISMA synthesis) come from the prisma-repo dataset. The figures in
Table 4 (expert round) come from `../delphi_ronda1/resultados/`.

## Relation to `owa_typology.py`

`owa_typology.py` is the reference implementation of v1.0.0 and keeps the original conventions:

- dimensions are numbered D1–D7 there; the correspondence is D1→D1, D2→D5, D3→D4, D4→D10, D5→D8, D6→D7, D7→D12;
- `PROFILES` is listed in the order produced by the equal-weight attitude score.

`analysis_a2.py` uses the codebook numbering and the lexicographic order of the thesis and of the manuscript.
It imports only the RIM and OWA functions from `owa_typology.py`, which are the same in both conventions.

`make_prisma_en.py` draws Supplementary Figure S1, the PRISMA 2020 flow. Its counts come from prisma-repo.
