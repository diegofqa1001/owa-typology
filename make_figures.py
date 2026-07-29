"""Figures for the Fuzzy-OWA typology paper. Okabe-Ito palette, white background, 300 dpi.
Run analysis_robustness.py first (produces robustness_results.json)."""
import json
import numpy as np
import matplotlib.pyplot as plt

from owa_typology import PROFILES, label_centroids, calibrate

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})
OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#999999', '#0072B2', '#D55E00', '#CC79A7']

cog = label_centroids()
base = calibrate()
names = list(base)
N = 7

# ---------------------------------------------------------------- Figure 1
dims = ['D1\nRisk\ntolerance', 'D2\nLoss\naversion', 'D3\nSelf-\nefficacy', 'D4\nAmbiguity\ntolerance',
        'D5\nHorizon', 'D6\nEmotional\nregulation', 'D7\nSocial\ninfluence']
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
fig, axes = plt.subplots(2, 4, figsize=(11, 6.4), subplot_kw=dict(polar=True))
for ax, name, c in zip(axes.flat, names, OI):
    vals = [cog[l] for l in PROFILES[name]]
    v = np.r_[vals, vals[0]]; a = np.r_[ang, ang[0]]
    ax.plot(a, v, color=c, lw=1.8); ax.fill(a, v, color=c, alpha=0.18)
    ax.set_ylim(0, 1); ax.set_xticks(ang); ax.set_xticklabels(dims, fontsize=5.6)
    ax.set_yticks([0.25, 0.5, 0.75]); ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=5)
    ax.set_title(f'P{names.index(name)+1} {name}\n$\\bar a$ = {base[name]["a"]:.3f}',
                 fontsize=8.5, pad=15, fontweight='bold')
    ax.grid(alpha=0.35, lw=0.5)
fig.tight_layout(h_pad=2.6)
fig.savefig('fig1_profiles_radar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(figsize=(7.2, 4.2))
j = np.arange(1, N + 1)
for name, c in zip(names, OI):
    ax.plot(j, base[name]['w'], marker='o', ms=4, lw=1.6, color=c,
            label=f'P{names.index(name)+1} {name} (orness = {base[name]["a"]:.3f})')
ax.set_xlabel('Ordered position $j$ (1 = largest value)')
ax.set_ylabel('OWA weight $w_j$')
ax.set_xticks(j)
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig('fig2_owa_weights.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
R = json.load(open('robustness_results.json'))
bp = R['breaking_point']
concs = sorted((float(k) for k in bp), reverse=True)
pct = [bp[str(int(c)) if float(c).is_integer() else str(c)]['pct_identical'] for c in concs]
stab = R['classification_stability']
sig = sorted(float(k) for k in stab)
rec = [stab[f'{s}']['overall_pct'] for s in sig]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(len(concs)), pct, marker='o', ms=5, lw=1.8, color=OI[5])
ax1.set_xticks(range(len(concs)))
ax1.set_xticklabels([f'{c:g}' for c in concs])
ax1.invert_xaxis()
ax1.set_xlabel('Dirichlet concentration (lower = more unequal weights)')
ax1.set_ylabel('Ordering identical to baseline (%)')
ax1.set_title('(a) Robustness of the profile ordering\nto dimension weighting', fontsize=9.5)
ax1.set_ylim(0, 105); ax1.grid(alpha=0.3, lw=0.5)

ax2.plot(sig, rec, marker='s', ms=5, lw=1.8, color=OI[3])
ax2.set_xlabel('Response noise $\\sigma$ (per dimension)')
ax2.set_ylabel('Correct profile recovered (%)')
ax2.set_title('(b) Stability of profile assignment\nunder measurement error', fontsize=9.5)
ax2.set_ylim(0, 105); ax2.set_xticks(sig); ax2.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig('fig3_robustness.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Figures written: fig1_profiles_radar.png, fig2_owa_weights.png, fig3_robustness.png')
