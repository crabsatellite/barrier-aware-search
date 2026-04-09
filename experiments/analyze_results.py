"""
Analyze experiment results and generate summary tables.

Usage:
    python analyze_results.py              # full analysis
    python analyze_results.py synthetic    # just synthetic benchmark
    python analyze_results.py crossopt     # just 30-seed cross-optimizer
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "jmlr_revision")


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Block 2C: Synthetic ablation
# ═══════════════════════════════════════════════════════════════════════════

def analyze_synthetic():
    results = load("synthetic_ablation.json")
    cells = results.get("_cells", {})
    if not cells:
        print("No synthetic results yet.")
        return

    CELL_ORDER = [
        ("rank_cross_binary", "C1+C2+C3"),
        ("rank_cross_soft",   "C1+C2"),
        ("rank_diag_binary",  "C1+C3"),
        ("rank_diag_soft",    "C1 only"),
        ("mse_cross_binary",  "C2+C3"),
        ("mse_cross_soft",    "C2 only"),
    ]

    print("\n" + "=" * 80)
    print("  SYNTHETIC ABLATION (Block 2C)")
    print("=" * 80)
    print(f"{'Cell':<25} {'Conditions':<12} {'full':>8} {'sep':>8} {'iso':>8}  Fisher(f>s)")
    print("-" * 80)

    for cell_label, conditions in CELL_ORDER:
        cell = cells.get(cell_label, {})
        rates = {}
        for v in ["full", "sep", "isotropic"]:
            drift = sum(1 for s in range(1, 31)
                        if f"{v}_seed{s}" in cell
                        and cell[f"{v}_seed{s}"]["final"].get("regime") == "B")
            total = sum(1 for s in range(1, 31) if f"{v}_seed{s}" in cell)
            rates[v] = (drift, total)

        fd, ft = rates["full"]
        sd, st = rates["sep"]
        p_str = "—"
        if ft > 0 and st > 0:
            table = [[fd, ft - fd], [sd, st - sd]]
            _, p = fisher_exact(table, alternative="greater")
            p_str = f"p={p:.4f}"
            if p < 0.05:
                p_str += " *"
            if p < 0.01:
                p_str = p_str.rstrip(" *") + " **"
            if p < 0.001:
                p_str = p_str.rstrip(" *") + " ***"

        print(f"{cell_label:<25} {conditions:<12} "
              f"{rates['full'][0]:>3}/{rates['full'][1]:<3}  "
              f"{rates['sep'][0]:>3}/{rates['sep'][1]:<3}  "
              f"{rates['isotropic'][0]:>3}/{rates['isotropic'][1]:<3}  "
              f"{p_str}")

    # phi_overhead comparison for key cell
    key_cell = cells.get("rank_cross_binary", {})
    if key_cell:
        print("\n--- phi_overhead distribution (rank_cross_binary) ---")
        for v in ["full", "sep", "isotropic"]:
            phis = [key_cell[f"{v}_seed{s}"]["final"]["phi_overhead"]
                    for s in range(1, 31) if f"{v}_seed{s}" in key_cell
                    and "phi_overhead" in key_cell[f"{v}_seed{s}"]["final"]]
            if phis:
                print(f"  {v:>10}: mean={np.mean(phis):.4f} "
                      f"std={np.std(phis):.4f} "
                      f"median={np.median(phis):.4f} "
                      f"[{np.min(phis):.4f}, {np.max(phis):.4f}]")

        # Mann-Whitney U test on phi values
        full_phi = [key_cell[f"full_seed{s}"]["final"]["phi_overhead"]
                    for s in range(1, 31) if f"full_seed{s}" in key_cell
                    and "phi_overhead" in key_cell[f"full_seed{s}"]["final"]]
        sep_phi = [key_cell[f"sep_seed{s}"]["final"]["phi_overhead"]
                   for s in range(1, 31) if f"sep_seed{s}" in key_cell
                   and "phi_overhead" in key_cell[f"sep_seed{s}"]["final"]]
        if full_phi and sep_phi:
            U, p_mw = mannwhitneyu(full_phi, sep_phi, alternative="greater")
            print(f"\n  Mann-Whitney U (full > sep): U={U:.0f}, p={p_mw:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Block 1A: 30-seed cross-optimizer
# ═══════════════════════════════════════════════════════════════════════════

def analyze_crossopt():
    results = load("crossopt_30seed.json")
    if not results:
        print("No crossopt results yet.")
        return

    print("\n" + "=" * 80)
    print("  30-SEED CROSS-OPTIMIZER (Block 1A)")
    print("=" * 80)

    variants = ["full", "sep", "isotropic"]
    seeds = list(range(1, 31))

    for v in variants:
        regimes = []
        rhos = []
        phis = []
        for s in seeds:
            key = f"{v}_seed{s}"
            if key in results:
                final = results[key]["final"]
                regimes.append(final.get("regime", "?"))
                if "r" in final:
                    rhos.append(final["r"])
                if "phi_sens" in final:
                    phis.append(final["phi_sens"])

        n = len(regimes)
        if n == 0:
            continue

        a = regimes.count("A")
        b = regimes.count("B")
        c = regimes.count("C")
        drift = a + b

        print(f"\n  {v} (n={n}):")
        print(f"    Regimes: A={a} B={b} C={c}")
        print(f"    Drift rate: {drift}/{n} ({100*drift/n:.0f}%)")
        if rhos:
            print(f"    rho: mean={np.mean(rhos):.4f} std={np.std(rhos):.4f}")
        if phis:
            print(f"    phi_sens: mean={np.mean(phis):.4f} std={np.std(phis):.4f}")

    # Fisher's exact test: full vs sep
    full_drift = sum(1 for s in seeds
                     if f"full_seed{s}" in results
                     and results[f"full_seed{s}"]["final"].get("regime") in ("A", "B"))
    full_total = sum(1 for s in seeds if f"full_seed{s}" in results)
    sep_drift = sum(1 for s in seeds
                    if f"sep_seed{s}" in results
                    and results[f"sep_seed{s}"]["final"].get("regime") in ("A", "B"))
    sep_total = sum(1 for s in seeds if f"sep_seed{s}" in results)

    if full_total > 0 and sep_total > 0:
        table = [[full_drift, full_total - full_drift],
                 [sep_drift, sep_total - sep_drift]]
        _, p = fisher_exact(table, alternative="greater")
        print(f"\n  Fisher's exact (full > sep): p = {p:.6f}")
        if p < 0.001:
            print("  *** Highly significant")
        elif p < 0.01:
            print("  ** Significant")
        elif p < 0.05:
            print("  * Significant")
        else:
            print("  Not significant at alpha=0.05")


# ═══════════════════════════════════════════════════════════════════════════
# Block 2A: SVM fairness
# ═══════════════════════════════════════════════════════════════════════════

def analyze_svm():
    results = load("svm_fairness.json")
    if not results:
        print("No SVM fairness results yet.")
        return

    print("\n" + "=" * 80)
    print("  SVM FAIRNESS DOMAIN (Block 2A)")
    print("=" * 80)

    variants = ["full", "sep", "isotropic"]
    seeds = list(range(1, 31))

    for v in variants:
        drift = sum(1 for s in seeds
                    if f"{v}_seed{s}" in results
                    and results[f"{v}_seed{s}"]["final"].get("regime") == "B")
        total = sum(1 for s in seeds if f"{v}_seed{s}" in results)
        if total:
            print(f"  {v:>10}: drift {drift}/{total} ({100*drift/total:.0f}%)")

    full_drift = sum(1 for s in seeds
                     if f"full_seed{s}" in results
                     and results[f"full_seed{s}"]["final"].get("regime") == "B")
    full_total = sum(1 for s in seeds if f"full_seed{s}" in results)
    sep_drift = sum(1 for s in seeds
                    if f"sep_seed{s}" in results
                    and results[f"sep_seed{s}"]["final"].get("regime") == "B")
    sep_total = sum(1 for s in seeds if f"sep_seed{s}" in results)

    if full_total > 0 and sep_total > 0:
        table = [[full_drift, full_total - full_drift],
                 [sep_drift, sep_total - sep_drift]]
        _, p = fisher_exact(table, alternative="greater")
        print(f"\n  Fisher's exact (full > sep): p = {p:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def full_analysis():
    analyze_synthetic()
    analyze_crossopt()
    analyze_svm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["all", "synthetic", "crossopt", "svm"])
    args = parser.parse_args()

    if args.mode == "synthetic":
        analyze_synthetic()
    elif args.mode == "crossopt":
        analyze_crossopt()
    elif args.mode == "svm":
        analyze_svm()
    else:
        full_analysis()


if __name__ == "__main__":
    main()
