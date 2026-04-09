"""
Seed 42 vs 137 truncation comparison — path-dependent compliance strategies.

Generates a 2x3 figure showing fundamentally different compliance mechanisms:
- Seed 42: dedicated compliance subspace (BICS), sensitivity = compliance lever
- Seed 137: embedded compliance in signal params, barrier trade-off (NP vs algebrization)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load data
with open(os.path.join(PROJECT_DIR, "data", "truncation_results.json")) as f:
    seed42 = json.load(f)

with open(os.path.join(PROJECT_DIR, "data", "truncation_seed_137.json")) as f:
    seed137 = json.load(f)

scales = [e["scale"] for e in seed42]

r42 = [e["r"] for e in seed42]
r137 = [e["r"] for e in seed137]

d42 = [e["np_density"] for e in seed42]
d137 = [e["np_density"] for e in seed137]

feas42 = [e["feasible"] for e in seed42]
feas137 = [e["feasible"] for e in seed137]

# barrier-specific: which barrier fails at each scale
np_pass_42 = [e["margins"]["natural_proof"]["passes"] for e in seed42]
alg_pass_42 = [e["margins"]["algebrization"]["passes"] for e in seed42]
np_pass_137 = [e["margins"]["natural_proof"]["passes"] for e in seed137]
alg_pass_137 = [e["margins"]["algebrization"]["passes"] for e in seed137]

NP_THRESH = 0.04

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# ── Row 1: Seed 42 ──────────────────────────────────────────────────────

# A1: r vs scale
ax = axes[0, 0]
ax.plot(scales, r42, 'o-', color='#1976d2', linewidth=1.5, markersize=5)
ax.axhline(0.5932, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.annotate('signal ceiling', xy=(0.35, 0.596), fontsize=7, color='gray')
ax.set_ylabel("Spearman r")
ax.set_title("A1. Prediction (seed 42)", fontweight='bold')
ax.set_ylim(0.1, 0.62)
ax.grid(True, alpha=0.3)
# shade feasible scale regions
for i in range(len(scales) - 1):
    if feas42[i]:
        ax.axvspan(scales[i], scales[i + 1], alpha=0.08, color='green')

# B1: NP density vs scale
ax = axes[0, 1]
ax.plot(scales, d42, 's-', color='#d32f2f', linewidth=1.5, markersize=5)
ax.axhline(NP_THRESH, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax.fill_between(scales, 0, NP_THRESH, alpha=0.06, color='green')
ax.set_ylabel("NP density")
ax.set_title("B1. NP density (seed 42)", fontweight='bold')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)

# C1: barrier status heatmap
ax = axes[0, 2]
status_42 = []
for i in range(len(scales)):
    if feas42[i]:
        status_42.append(2)  # all pass
    elif not np_pass_42[i]:
        status_42.append(0)  # NP fails
    elif not alg_pass_42[i]:
        status_42.append(1)  # alg fails
    else:
        status_42.append(0)
bar_colors_42 = ['#d32f2f' if s == 0 else '#ff9800' if s == 1 else '#4caf50' for s in status_42]
ax.bar(scales, [1]*len(scales), width=0.08, color=bar_colors_42, alpha=0.8)
ax.set_yticks([])
ax.set_title("C1. Barrier status (seed 42)", fontweight='bold')
ax.set_xlabel("Sensitivity scale")
# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d32f2f', alpha=0.8, label='NP fails'),
    Patch(facecolor='#ff9800', alpha=0.8, label='Alg fails'),
    Patch(facecolor='#4caf50', alpha=0.8, label='All pass'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

# ── Row 2: Seed 137 ─────────────────────────────────────────────────────

# A2: r vs scale
ax = axes[1, 0]
ax.plot(scales, r137, 'o-', color='#1976d2', linewidth=1.5, markersize=5)
ax.axhline(0.5932, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.annotate('signal ceiling', xy=(0.35, 0.5935), fontsize=7, color='gray')
ax.set_xlabel("Sensitivity parameter scale")
ax.set_ylabel("Spearman r")
ax.set_title("A2. Prediction (seed 137)", fontweight='bold')
ax.set_ylim(0.1, 0.62)
ax.grid(True, alpha=0.3)
# no feasible regions for seed 137

# B2: NP density vs scale
ax = axes[1, 1]
ax.plot(scales, d137, 's-', color='#d32f2f', linewidth=1.5, markersize=5)
ax.axhline(NP_THRESH, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax.fill_between(scales, 0, NP_THRESH, alpha=0.06, color='green')
ax.set_xlabel("Sensitivity parameter scale")
ax.set_ylabel("NP density")
ax.set_title("B2. NP density (seed 137)", fontweight='bold')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)

# C2: barrier status heatmap
ax = axes[1, 2]
status_137 = []
for i in range(len(scales)):
    if feas137[i]:
        status_137.append(2)
    elif not np_pass_137[i]:
        status_137.append(0)
    elif not alg_pass_137[i]:
        status_137.append(1)
    else:
        status_137.append(0)
bar_colors_137 = ['#d32f2f' if s == 0 else '#ff9800' if s == 1 else '#4caf50' for s in status_137]
ax.bar(scales, [1]*len(scales), width=0.08, color=bar_colors_137, alpha=0.8)
ax.set_yticks([])
ax.set_title("C2. Barrier status (seed 137)", fontweight='bold')
ax.set_xlabel("Sensitivity scale")
ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

# ── Annotations ──────────────────────────────────────────────────────────

# seed 42 summary
axes[0, 0].annotate(
    'Phase transition:\nr crashes 0.59→0.17\nat scale 0.05',
    xy=(0.05, 0.19), xytext=(0.25, 0.30),
    fontsize=7, color='#d32f2f',
    arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=0.8),
)

# seed 137 summary
axes[1, 0].annotate(
    'No phase transition:\nr flat 0.596→0.597',
    xy=(0.5, 0.5963), xytext=(0.15, 0.40),
    fontsize=7, color='#1976d2',
    arrowprops=dict(arrowstyle='->', color='#1976d2', lw=0.8),
)

fig.suptitle(
    "Path-dependent compliance: seed 42 (dedicated BICS) vs seed 137 (embedded compliance)",
    fontsize=13, fontweight='bold', y=0.98,
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(PROJECT_DIR, "figures", "bics_seed_comparison.pdf")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")
