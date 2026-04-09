"""
Basin topology figure — 5-seed endpoint map + interpolation connectivity matrix.

Three compliance strategy types discovered:
  Type A: Dedicated BICS (seed 42)          — 70% sens, 81% cross-term, feasible
  Type B: Embedded (seeds 137, 256, 1024)   — 19-24% sens, 13-15% cross-term, borderline
  Type C: Pure signal (seed 512)            — 0.4% sens, 0.9% cross-term, feasible, best r
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load endpoint data ──────────────────────────────────────────────────

seeds = [42, 137, 256, 512, 1024]
endpoints = {}
for s in seeds:
    path = os.path.join(PROJECT_DIR, "data", f"endpoint_seed_{s}.json")
    with open(path) as f:
        endpoints[s] = json.load(f)

# ── Load interpolation data ─────────────────────────────────────────────

pairs = [(42, 137), (42, 256), (42, 512), (137, 256), (137, 512), (256, 512)]
interp = {}
for a, b in pairs:
    path = os.path.join(PROJECT_DIR, "data", f"interpolation_{a}_{b}.json")
    with open(path) as f:
        interp[(a, b)] = json.load(f)

# ── Figure: 2x2 layout ─────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))

# Panel A: Endpoint scatter — cross_term_frac vs sensitivity
ax1 = fig.add_subplot(2, 2, 1)
for s in seeds:
    e = endpoints[s]
    color = '#1976d2' if s == 42 else '#ff9800' if s in (137, 256, 1024) else '#4caf50'
    marker = 'D' if s == 42 else 's' if s in (137, 256) else '*' if s == 512 else '^'
    size = 120 if s in (42, 512) else 80
    ax1.scatter(e["sensitivity_importance"], e["cross_term_frac"],
                c=color, marker=marker, s=size, zorder=5, edgecolors='black', linewidth=0.5)
    label = f'seed {s}\nr={e["r"]:.4f}'
    if e["feasible"]:
        label += ' (F)'
    ax1.annotate(label, (e["sensitivity_importance"], e["cross_term_frac"]),
                 textcoords="offset points", xytext=(12, 5), fontsize=8)

ax1.set_xlabel("Sensitivity importance", fontsize=10)
ax1.set_ylabel("Cross-term fraction (quadratic mass in sensitivity)", fontsize=10)
ax1.set_title("A. Endpoint fingerprint", fontweight='bold', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 0.85)
ax1.set_ylim(-0.05, 0.95)

# type labels
ax1.annotate('Type A:\nDedicated BICS', xy=(0.65, 0.75), fontsize=9, color='#1976d2',
             fontstyle='italic', ha='center')
ax1.annotate('Type B:\nEmbedded', xy=(0.25, 0.18), fontsize=9, color='#ff9800',
             fontstyle='italic', ha='center')
ax1.annotate('Type C:\nPure Signal', xy=(0.05, -0.02), fontsize=9, color='#4caf50',
             fontstyle='italic', ha='center', va='top')

# Panel B: r vs sensitivity overhead — shows inverse correlation
ax2 = fig.add_subplot(2, 2, 2)
sens_vals = [endpoints[s]["sensitivity_importance"] for s in seeds]
r_vals = [endpoints[s]["r"] for s in seeds]
feas_vals = [endpoints[s]["feasible"] for s in seeds]

for i, s in enumerate(seeds):
    color = '#1976d2' if s == 42 else '#ff9800' if s in (137, 256, 1024) else '#4caf50'
    marker = 'D' if s == 42 else 's' if s in (137, 256) else '*' if s == 512 else '^'
    size = 150 if s in (42, 512) else 100
    edge = 'black' if feas_vals[i] else '#d32f2f'
    lw = 0.5 if feas_vals[i] else 2.0
    ax2.scatter(sens_vals[i], r_vals[i], c=color, marker=marker, s=size,
                zorder=5, edgecolors=edge, linewidth=lw)
    label = f'seed {s}'
    ax2.annotate(label, (sens_vals[i], r_vals[i]),
                 textcoords="offset points", xytext=(10, -5), fontsize=9)

ax2.set_xlabel("Sensitivity importance (compliance overhead)", fontsize=10)
ax2.set_ylabel("Spearman r (prediction quality)", fontsize=10)
ax2.set_title("B. Overhead vs performance", fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3)
# trend line
z = np.polyfit(sens_vals, r_vals, 1)
x_line = np.linspace(-0.05, 0.75, 50)
ax2.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.5, linewidth=1)
ax2.annotate(f'slope = {z[0]:.4f}', xy=(0.35, 0.5955), fontsize=8, color='gray')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
           markeredgewidth=0.5, markersize=8, label='Feasible'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='#d32f2f',
           markeredgewidth=2, markersize=8, label='Infeasible'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='lower left')

# Panel C: Connectivity matrix heatmap — min r along interpolation
ax3 = fig.add_subplot(2, 2, 3)
n = len(seeds)
conn_matrix = np.full((n, n), np.nan)
for (a, b), data in interp.items():
    rs = [e["r"] for e in data]
    min_r = min(rs)
    i, j = seeds.index(a), seeds.index(b)
    conn_matrix[i, j] = min_r
    conn_matrix[j, i] = min_r
# diagonal = self
for i in range(n):
    conn_matrix[i, i] = endpoints[seeds[i]]["r"]

im = ax3.imshow(conn_matrix, cmap='RdYlGn', vmin=-0.2, vmax=0.6, aspect='auto')
ax3.set_xticks(range(n))
ax3.set_yticks(range(n))
ax3.set_xticklabels([f's{s}' for s in seeds], fontsize=9)
ax3.set_yticklabels([f's{s}' for s in seeds], fontsize=9)
ax3.set_title("C. Connectivity (min r on path)", fontweight='bold', fontsize=11)

# annotate values
for i in range(n):
    for j in range(n):
        val = conn_matrix[i, j]
        color = 'white' if val < 0.1 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax3, shrink=0.8, label='min Spearman r')

# Panel D: Connectivity matrix — max NP density along interpolation
ax4 = fig.add_subplot(2, 2, 4)
density_matrix = np.full((n, n), np.nan)
for (a, b), data in interp.items():
    ds = [e["np_density"] for e in data]
    max_d = max(ds)
    i, j = seeds.index(a), seeds.index(b)
    density_matrix[i, j] = max_d
    density_matrix[j, i] = max_d
for i in range(n):
    density_matrix[i, i] = endpoints[seeds[i]]["np_density"]

im2 = ax4.imshow(density_matrix, cmap='RdYlGn_r', vmin=0, vmax=1.0, aspect='auto')
ax4.set_xticks(range(n))
ax4.set_yticks(range(n))
ax4.set_xticklabels([f's{s}' for s in seeds], fontsize=9)
ax4.set_yticklabels([f's{s}' for s in seeds], fontsize=9)
ax4.set_title("D. Barrier wall (max NP density on path)", fontweight='bold', fontsize=11)

for i in range(n):
    for j in range(n):
        val = density_matrix[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold', color=color)

plt.colorbar(im2, ax=ax4, shrink=0.8, label='max NP density')

fig.suptitle(
    "Basin topology: three compliance strategies in barrier-constrained search",
    fontsize=14, fontweight='bold', y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(PROJECT_DIR, "figures", "basin_topology.pdf")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")
