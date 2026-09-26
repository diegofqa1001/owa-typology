"""Figuras 3.1, 3.2 y 3.3 de la monografía (Capítulo 3), rótulos en español y coma decimal.

Datos: parámetros trapezoidales de las etiquetas (Tabla 3.1), vectores lingüísticos de los
ocho perfiles (Tabla 3.2) y orness v1 por centroides (Tabla 3.3). Paleta Okabe-Ito, fondo
blanco, 300 dpi.
"""
import locale
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

OI = ['#0072B2', '#56B4E9', '#009E73', '#F0E442', '#E69F00', '#D55E00', '#CC79A7', '#000000']
coma = FuncFormatter(lambda x, _: f'{x:.1f}'.replace('.', ','))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
OUT = 'tesis_cap3/figuras/'

# ---------- Figura 3.1: funciones de pertenencia ----------
ET = {'VL (Muy bajo)': (0, 0, .10, .25), 'L (Bajo)': (.10, .25, .25, .40), 'M (Moderado)': (.30, .45, .55, .70),
      'H (Alto)': (.60, .75, .75, .90), 'VH (Muy alto)': (.75, .90, 1.0, 1.0)}
col = [OI[0], OI[1], OI[2], OI[4], OI[5]]
def trap(x, a, b, c, d):
    y = np.zeros_like(x)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    if b > a: y = np.where((x > a) & (x < b), (x - a) / (b - a), y)
    if d > c: y = np.where((x > c) & (x < d), (d - x) / (d - c), y)
    return y
x = np.linspace(0, 1, 1001)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
for (k, p), c in zip(ET.items(), col):
    y = trap(x, *p); ax.plot(x, y, color=c, lw=2.2, label=k); ax.fill_between(x, y, color=c, alpha=.10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
ax.xaxis.set_major_formatter(coma); ax.yaxis.set_major_formatter(coma)
ax.set_xlabel('Valor normalizado de la dimensión conductual'); ax.set_ylabel('Grado de pertenencia μ(x)')
ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + 'fig_3_1_pertenencia.png', dpi=300, facecolor='white'); plt.close(fig)

# ---------- Figura 3.2: perfiles sobre el eje del orness (v1) ----------
PER = ['Guardián', 'Centinela', 'Pragmático', 'Analista', 'Estratega', 'Aventurero', 'Innovador', 'Visionario']
ORN = [0.158, 0.257, 0.503, 0.600, 0.647, 0.693, 0.738, 0.865]
colp = [OI[0], OI[1], OI[7], OI[2], OI[3], OI[4], OI[5], OI[6]]
fig, ax = plt.subplots(figsize=(10, 3.3), dpi=300)
ax.axhline(0, color='black', lw=1.8, zorder=1)
ax.axvline(0.5, color='grey', ls='--', lw=1.5, zorder=0)
ax.text(0.5, 1.02, 'orness = 0,5\n(régimen: media)', ha='center', va='center', color='grey', fontsize=10)
ax.text(0.02, 0.88, 'AND (conjuntivo, pesimista)', color=OI[0], fontsize=10.5, va='center')
ax.text(0.98, 0.88, 'OR (disyuntivo, optimista)', color=OI[6], fontsize=10.5, va='center', ha='right')
arriba = {'Guardián', 'Pragmático', 'Estratega', 'Innovador'}
for n, o, c in zip(PER, ORN, colp):
    ax.scatter(o, 0, s=260, color=c, edgecolor='black', zorder=3)
    s = 1 if n in arriba else -1
    ax.plot([o, o], [0.04 * s, 0.33 * s], color='grey', lw=1, zorder=2)
    ax.text(o, 0.5 * s, f'{n}\n({o:.3f})'.replace('.', ','), ha='center', va='center', fontsize=10.5)
ax.set_xlim(0, 1); ax.set_ylim(-1.1, 1.15); ax.set_yticks([])
for s in ('left', 'top', 'right'): ax.spines[s].set_visible(False)
ax.xaxis.set_major_formatter(coma)
ax.set_xlabel('Grado actitudinal — orness α(μ)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT + 'fig_3_2_eje_orness.png', dpi=300, facecolor='white'); plt.close(fig)

# ---------- Figura 3.3: mapa de calor perfil × dimensión ----------
VEC = {'Guardián': 'VL VH VL VL VL VL VH', 'Centinela': 'L H M L M L H', 'Pragmático': 'M M M M M M M',
       'Analista': 'M M VH H H H VL', 'Estratega': 'H M H M H H L', 'Aventurero': 'H L H M H M H',
       'Innovador': 'VH L VH H VH H L', 'Visionario': 'VH VL VH VH VH VH VL'}
VAL = {'VL': 0.09, 'L': 0.25, 'M': 0.50, 'H': 0.75, 'VH': 0.91}  # centroides de las etiquetas
DIMS = ['Toler.\nriesgo (D1)', 'Aversión\npérdida (D5)', 'Auto-\neficacia (D4)', 'Toler.\nambig. (D10)',
        'Horizonte\n(D8)', 'Regul.\nemoc. (D7)', 'Influ.\nsocial (D12)']
M = np.array([[VAL[e] for e in VEC[p].split()] for p in PER])
cmap = LinearSegmentedColormap.from_list('oi_azul', ['#FFFFFF', '#56B4E9', '#0072B2'])
fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i, p in enumerate(PER):
    for j, e in enumerate(VEC[p].split()):
        ax.text(j, i, e, ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if VAL[e] >= 0.7 else 'black')
ax.set_xticks(range(7)); ax.set_xticklabels(DIMS, fontsize=8.5)
ax.set_yticks(range(8)); ax.set_yticklabels(PER)
for s in ax.spines.values(): s.set_visible(True)
ax.set_title('Vectores lingüísticos difusos de los ocho perfiles (D5 y D12 codificadas inversamente)', fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.yaxis.set_major_formatter(coma)
cb.set_label('Intensidad lingüística (VL→VH)', fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT + 'fig_3_3_mapa_calor.png', dpi=300, facecolor='white'); plt.close(fig)
print('OK figuras 3.1–3.3')
