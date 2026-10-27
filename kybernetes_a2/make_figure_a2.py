#!/usr/bin/env python3
"""Figure 1 of the Kybernetes manuscript: (a) directional profile x dimension
matrix in rank order, core and modulating blocks separated; (b) OWA weight
vectors induced by the octile anchors through the RIM quantifier.
Navy-and-grey palette, white background, 300 dpi.
Run analysis_a2.py first (reads results/a2_robustness.json)."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = json.load(open(os.path.join(ROOT, "results", "a2_robustness.json"), encoding="utf-8"))

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans",
                     "font.size": 8})
NAVY = "#1F3A5F"
CMAP = LinearSegmentedColormap.from_list("ng", ["#FFFFFF", "#9AACC2", NAVY])
COG = R["cog_labels"]
NAMES = list(R["profiles"])
COLS = ["D1", "1−D5", "D4", "D10", "D8", "D7", "1−D12"]
M = []
for n in NAMES:
    v = [COG[l] for l in R["profiles"][n]["vector"]]
    v[1], v[6] = 1 - v[1], 1 - v[6]
    M.append(v)
M = np.array(M)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
im = a1.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
for i in range(8):
    for j in range(7):
        a1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                color="white" if M[i, j] > 0.62 else "black")
a1.axvline(1.5, color="black", lw=1.4)
a1.set_xticks(range(7), COLS, fontsize=7)
a1.set_yticks(range(8), [f"{k + 1} {n}" for k, n in enumerate(NAMES)], fontsize=7)
a1.set_title("(a) Directional profile vectors\ncore | modulating dimensions", fontsize=8)
a1.tick_params(length=0)
cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)

greys = ["#000000", "#3F3F3F", "#8C8C8C", "#9AACC2", "#4F6D8F", "#1F3A5F", "#999999", "#8A7D72"]
styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
x = np.arange(1, 8)
for k, n in enumerate(NAMES):
    p = R["profiles"][n]
    a2.plot(x, p["weights"], styles[k], marker=markers[k], ms=3.5, lw=1.2, color=greys[k],
            label=f"{n} ({p['octile_anchor']:.4f})")
a2.set_xlabel("Ordered position j (1 = best criterion score)")
a2.set_ylabel("OWA weight $w_j$")
a2.set_xticks(x)
a2.set_ylim(0, 0.9)
a2.spines[["top", "right"]].set_visible(False)
a2.legend(fontsize=5.8, frameon=False, title="Profile (orness)", title_fontsize=6.2, loc="upper center")
a2.set_title("(b) Weights induced by the octile anchors", fontsize=8)
fig.tight_layout()
out = os.path.join(HERE, "figure1_a2.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("->", out)
