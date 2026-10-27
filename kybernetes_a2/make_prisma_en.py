"""Supplementary Figure S1: PRISMA 2020 flow diagram (English), counts from prisma-repo / thesis Fig. 3.4."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, GREY = "#1F3A5F", "#8C8C8C"
fig, ax = plt.subplots(figsize=(7.2, 8.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
def box(x, y, w, h, t, c=NAVY, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc="white", ec=c, lw=1.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
for y, h, lab in [(10.0, 1.6, "Identification"), (5.9, 3.8, "Screening"), (4.0, 1.5, "Eligibility"), (0.2, 3.4, "Synthesis")]:
    ax.add_patch(FancyBboxPatch((0.1, y), 0.45, h, boxstyle="round,pad=0.02", fc=NAVY, ec=NAVY))
    ax.text(0.325, y + h / 2, lab, rotation=90, ha="center", va="center", color="white", fontsize=8, weight="bold")
box(0.9, 10.1, 4.0, 1.4, "Records identified\nScopus n = 438; Web of Science n = 289\nTotal n = 727")
box(5.6, 10.1, 4.0, 1.4, "Duplicates removed\n(normalised title, similarity ≥ 0.90)\nn = 167", GREY)
box(0.9, 7.9, 4.0, 1.6, "Unique records screened\n(title and abstract)\nn = 560")
box(5.6, 7.9, 4.0, 1.6, "Records excluded n = 217\n(161 by criteria E1–E7; 48 uncertain,\nexcluded by protocol; 8 included records\nexcluded after validation)", GREY, 7.0)
box(0.9, 6.0, 4.0, 1.5, "Records included (abstract level)\nn = 343\nModel-assisted screening; second pass\non a 20% sample, κ = 0.799")
box(5.6, 6.0, 4.0, 1.5, "Human verification by two reviewers\nof all screening decisions\nand of eligibility", GREY)
box(0.9, 4.1, 4.0, 1.3, "Eligibility assessment\nn = 343", GREY)
box(0.9, 0.3, 4.0, 3.2, "Thematic synthesis (abstract level)\nn = 343 coded\nCodebook fixed in advance:\n14 candidate dimensions (D1–D14)\nTwo model-assisted coding passes\nκ per dimension 0.76–0.96\n149 disagreements adjudicated\n→ 7 dimensions retained")
box(5.6, 0.3, 4.0, 3.2, "Retained dimensions (n abstracts)\nD1 Risk tolerance: 189\nD5 Loss aversion: 153\nD12 Social influence: 96\nD7 Emotional regulation: 48\nD8 Horizon: 12\nD4 Self-efficacy: 9\nD10 Ambiguity tolerance: 6")
arrow(4.9, 10.8, 5.6, 10.8); arrow(2.9, 10.1, 2.9, 9.5); arrow(4.9, 8.7, 5.6, 8.7); arrow(2.9, 7.9, 2.9, 7.5)
arrow(2.9, 6.0, 2.9, 5.4); arrow(2.9, 4.1, 2.9, 3.5); arrow(4.9, 1.9, 5.6, 1.9)
out = os.path.join(HERE, "figureS1_prisma.png"); fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print(out)
