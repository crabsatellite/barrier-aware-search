"""
Attack experiments: prove signal/calibration decomposition is structurally inescapable.

Experiments:
  cross_only         — disable all self-quadratic (m_i²), allow only cross-terms
  label_random       — shuffle circuit size labels, re-run barrier-aware search
  perturb            — perturb sensitivity module (sign flip, scale, noise), evaluate
  phase_diagram      — plot r / sensitivity weight / feasibility vs generation

Usage: python attacks.py <experiment>
  python attacks.py cross_only
  python attacks.py label_random
  python attacks.py perturb
  python attacks.py phase_diagram
"""

import json
import logging
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "attacks.log")),
    ],
)
log = logging.getLogger(__name__)

from mechanism import (
    MEASURE_NAMES, D, Q_DIM, GENOME_DIM, W_SLICE, Q_SLICE, A_SLICE, T_SLICE,
    load_data, load_best_genome, evaluate_genome,
)
from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
from barriers import check_all_barriers_batch, RelativizationChecker
from search import CMAES
from analyze_candidates import feature_importance

RESULT_PATH = os.path.join(PROJECT_DIR, "data", "attack_results.json")


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {}


def save_results(data):
    with open(RESULT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def setup():
    measures, quad, targets = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    knuth_eval = MeasureEvaluator(measures, quad, device)
    target_ranks = precompute_target_ranks(targets, device)

    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        cfg = json.load(f)

    data = np.load(os.path.join(PROJECT_DIR, "data", "search_features.npz"))
    random_eval = MeasureEvaluator(data["random_measures"], data["random_quad"], device)
    rel_checker = RelativizationChecker(
        data["knuth_bits"],
        poly_degree=cfg["barriers"]["relativization"]["poly_degree"],
        n_samples=cfg["barriers"]["relativization"]["n_samples"],
    )
    return (measures, quad, targets, knuth_eval, random_eval,
            target_ranks, rel_checker, cfg, data)


def compute_self_quad_indices():
    """Return list of Q_SLICE-relative indices for self-quadratic terms (m_i²)."""
    indices = []
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == j:
                indices.append(k)
            k += 1
    return indices


def get_sensitivity_quad_indices():
    """Return Q_SLICE-relative indices for all quadratic terms involving sensitivity."""
    si = MEASURE_NAMES.index("sensitivity")
    indices = []
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                indices.append(k)
            k += 1
    return indices


def analyze_genome_quick(genome, measures, quad, targets):
    """Quick analysis: full_r, top-3 ablation, feature importance."""
    full_r = evaluate_genome(genome, measures, quad, targets)
    imp = feature_importance(genome, D)

    drops = {}
    for fi, fname in enumerate(MEASURE_NAMES):
        ablated = genome.copy()
        ablated[fi] = 0.0
        k = 0
        for i in range(D):
            for j in range(i, D):
                if i == fi or j == fi:
                    ablated[D + k] = 0.0
                k += 1
        ablated[D + Q_DIM + fi] = 0.0
        r_abl = evaluate_genome(ablated, measures, quad, targets)
        drops[fname] = full_r - r_abl

    return {
        "full_r": full_r,
        "importance": {MEASURE_NAMES[i]: float(imp[i]) for i in range(D)},
        "top_signal": sorted(drops.items(), key=lambda x: -x[1])[:3],
        "top_importance": sorted(
            [(MEASURE_NAMES[i], float(imp[i])) for i in range(D)],
            key=lambda x: -x[1])[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Attack 1: Cross-interaction only (no self-quadratic)
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_only(setup_data):
    """CMA-ES search with all self-quadratic terms (m_i²) zeroed after each ask()."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    log.info("=== ATTACK: Cross-interaction only (no self-quadratic) ===")
    self_quad = compute_self_quad_indices()
    log.info("  Zeroing %d self-quadratic indices: %s", len(self_quad), self_quad)

    t0 = time.time()
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]

    es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=100, seed=42)

    best_r = 0.0
    best_genome = None
    convergence = []
    stall_count = 0
    prev_best = 0.0

    for gen in range(120):
        candidates = es.ask()

        # zero out all self-quadratic terms
        for k in self_quad:
            candidates[:, D + k] = 0.0

        mu_knuth = knuth_eval.evaluate_batch(candidates)
        mu_random = random_eval.evaluate_batch(candidates)
        corr = spearman_batch_gpu(mu_knuth, target_ranks)

        barrier_pass, barrier_details = check_all_barriers_batch(
            mu_knuth, mu_random, candidates, rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_violations = np.zeros(100, dtype=np.int32)
        for i in range(100):
            if not barrier_pass[i]:
                n_violations[i] = sum(
                    1 for v in barrier_details[i].values() if not v["passes"])

        l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
        fitness = -corr + penalty * n_violations + l2_lambda * l2_norm
        es.tell(candidates, fitness)

        if es.should_restart():
            es.restart()

        idx_best = corr.argmax()
        if corr[idx_best] > best_r:
            best_r = float(corr[idx_best])
            best_genome = candidates[idx_best].copy()

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall_count = 0
        else:
            stall_count += 1

        feas_corrs = corr[barrier_pass.astype(bool)]
        feas_best = float(feas_corrs.max()) if len(feas_corrs) > 0 else 0.0

        if gen % 50 == 0 or gen == 119:
            convergence.append({
                "gen": gen + 1, "best_r": best_r, "feas_r": feas_best,
                "n_feasible": int(barrier_pass.sum()),
            })
            log.info("  gen %4d | best_r %.4f  feas %.4f | %d/100 | sigma %.2e",
                     gen + 1, best_r, feas_best, int(barrier_pass.sum()), es.sigma)

        if stall_count >= 100:
            log.info("  Early stop at gen %d", gen + 1)
            break

    analysis = analyze_genome_quick(best_genome, measures, quad, targets)
    elapsed = time.time() - t0

    result = {
        "method": "cross_only_search",
        "description": "All self-quadratic (m_i^2) zeroed, only cross-terms allowed",
        "n_self_quad_zeroed": len(self_quad),
        "best_r_gpu": best_r,
        "analysis": analysis,
        "convergence": convergence,
        "elapsed_s": elapsed,
        "genome": best_genome.tolist(),
    }

    log.info("  Full r (scipy) = %.4f", analysis["full_r"])
    log.info("  Top signal: %s",
             ", ".join(f"{n}={d:.4f}" for n, d in analysis["top_signal"]))
    log.info("  Top importance: %s",
             ", ".join(f"{n}={w:.3f}" for n, w in analysis["top_importance"]))
    log.info("  Done in %.1f min", elapsed / 60)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Attack 2: Label randomization
# ═══════════════════════════════════════════════════════════════════════════

def run_label_random(setup_data, seed=123):
    """Shuffle circuit size labels (preserving distribution), re-run search."""
    measures, quad, targets, knuth_eval, random_eval, _, rel_checker, cfg, _ = setup_data

    log.info("=== ATTACK: Label randomization (seed=%d) ===", seed)
    rng = np.random.RandomState(seed)
    shuffled = targets.copy()
    rng.shuffle(shuffled)

    # verify distribution preserved
    from collections import Counter
    orig_dist = Counter(targets.astype(int))
    shuf_dist = Counter(shuffled.astype(int))
    assert orig_dist == shuf_dist, "Distribution changed after shuffle!"
    log.info("  Label distribution preserved: %d unique sizes", len(orig_dist))

    # compute correlation between original and shuffled
    r_labels, _ = spearmanr(targets, shuffled)
    log.info("  Original-shuffled label correlation: %.4f", r_labels)

    # re-compute target ranks for shuffled labels
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shuffled_ranks = precompute_target_ranks(shuffled, device)

    t0 = time.time()
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]

    es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=100, seed=42)

    best_r = 0.0
    best_genome = None
    convergence = []
    stall_count = 0
    prev_best = 0.0

    for gen in range(120):
        candidates = es.ask()
        mu_knuth = knuth_eval.evaluate_batch(candidates)
        mu_random = random_eval.evaluate_batch(candidates)
        corr = spearman_batch_gpu(mu_knuth, shuffled_ranks)

        barrier_pass, barrier_details = check_all_barriers_batch(
            mu_knuth, mu_random, candidates, rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_violations = np.zeros(100, dtype=np.int32)
        for i in range(100):
            if not barrier_pass[i]:
                n_violations[i] = sum(
                    1 for v in barrier_details[i].values() if not v["passes"])

        l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
        fitness = -corr + penalty * n_violations + l2_lambda * l2_norm
        es.tell(candidates, fitness)

        if es.should_restart():
            es.restart()

        idx_best = corr.argmax()
        if corr[idx_best] > best_r:
            best_r = float(corr[idx_best])
            best_genome = candidates[idx_best].copy()

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall_count = 0
        else:
            stall_count += 1

        feas_corrs = corr[barrier_pass.astype(bool)]
        feas_best = float(feas_corrs.max()) if len(feas_corrs) > 0 else 0.0

        if gen % 50 == 0 or gen == 119:
            convergence.append({
                "gen": gen + 1, "best_r": best_r, "feas_r": feas_best,
                "n_feasible": int(barrier_pass.sum()),
            })
            log.info("  gen %4d | best_r %.4f  feas %.4f | %d/100 | sigma %.2e",
                     gen + 1, best_r, feas_best, int(barrier_pass.sum()), es.sigma)

        if stall_count >= 100:
            log.info("  Early stop at gen %d", gen + 1)
            break

    elapsed = time.time() - t0

    # evaluate best genome against ORIGINAL targets for comparison
    r_vs_original = evaluate_genome(best_genome, measures, quad, targets)
    r_vs_shuffled = evaluate_genome(best_genome, measures, quad, shuffled)

    result = {
        "method": "label_randomization",
        "seed": seed,
        "label_corr_original_shuffled": float(r_labels),
        "best_r_gpu_vs_shuffled": best_r,
        "r_vs_original_targets": r_vs_original,
        "r_vs_shuffled_targets": r_vs_shuffled,
        "convergence": convergence,
        "elapsed_s": elapsed,
    }

    log.info("  r vs shuffled labels (scipy) = %.4f", r_vs_shuffled)
    log.info("  r vs original labels (scipy) = %.4f", r_vs_original)
    log.info("  Done in %.1f min", elapsed / 60)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Attack 3: Perturb sensitivity calibration module
# ═══════════════════════════════════════════════════════════════════════════

def run_perturb(setup_data):
    """Perturb sensitivity-related genome weights and measure r degradation."""
    measures, quad, targets, _, _, _, _, _, _ = setup_data
    genome = load_best_genome()

    log.info("=== ATTACK: Perturb sensitivity calibration module ===")

    si = MEASURE_NAMES.index("sensitivity")
    full_r = evaluate_genome(genome, measures, quad, targets)
    log.info("  Baseline r = %.4f", full_r)

    sens_quad_indices = get_sensitivity_quad_indices()
    log.info("  Sensitivity quadratic indices (Q-relative): %s", sens_quad_indices)

    results = {"baseline_r": full_r, "perturbations": {}}

    # --- Perturbation 1: Sign flip all sensitivity quadratic weights ---
    g = genome.copy()
    for k in sens_quad_indices:
        g[D + k] = -g[D + k]
    r = evaluate_genome(g, measures, quad, targets)
    results["perturbations"]["sign_flip_quad"] = {
        "r": r, "drop": full_r - r, "drop_pct": 100 * (full_r - r) / full_r,
        "description": "Negate all sensitivity quadratic weights",
    }
    log.info("  Sign flip quad:   r = %.4f  (drop %.4f = %.1f%%)", r, full_r - r,
             100 * (full_r - r) / full_r)

    # --- Perturbation 2: Sign flip sensitivity linear weight ---
    g = genome.copy()
    g[si] = -g[si]
    r = evaluate_genome(g, measures, quad, targets)
    results["perturbations"]["sign_flip_linear"] = {
        "r": r, "drop": full_r - r, "drop_pct": 100 * (full_r - r) / full_r,
        "description": "Negate sensitivity linear weight only",
    }
    log.info("  Sign flip linear: r = %.4f  (drop %.4f = %.1f%%)", r, full_r - r,
             100 * (full_r - r) / full_r)

    # --- Perturbation 3: Sign flip ALL sensitivity terms ---
    g = genome.copy()
    g[si] = -g[si]
    for k in sens_quad_indices:
        g[D + k] = -g[D + k]
    g[D + Q_DIM + si] = -g[D + Q_DIM + si]
    r = evaluate_genome(g, measures, quad, targets)
    results["perturbations"]["sign_flip_all"] = {
        "r": r, "drop": full_r - r, "drop_pct": 100 * (full_r - r) / full_r,
        "description": "Negate all sensitivity-related weights",
    }
    log.info("  Sign flip all:    r = %.4f  (drop %.4f = %.1f%%)", r, full_r - r,
             100 * (full_r - r) / full_r)

    # --- Perturbation 4: Scale sensitivity quad ×2 ---
    g = genome.copy()
    for k in sens_quad_indices:
        g[D + k] *= 2.0
    r = evaluate_genome(g, measures, quad, targets)
    results["perturbations"]["scale_2x_quad"] = {
        "r": r, "drop": full_r - r, "drop_pct": 100 * (full_r - r) / full_r,
        "description": "Double all sensitivity quadratic weights",
    }
    log.info("  Scale 2x quad:    r = %.4f  (drop %.4f = %.1f%%)", r, full_r - r,
             100 * (full_r - r) / full_r)

    # --- Perturbation 5: Scale sensitivity quad ×0.5 ---
    g = genome.copy()
    for k in sens_quad_indices:
        g[D + k] *= 0.5
    r = evaluate_genome(g, measures, quad, targets)
    results["perturbations"]["scale_half_quad"] = {
        "r": r, "drop": full_r - r, "drop_pct": 100 * (full_r - r) / full_r,
        "description": "Halve all sensitivity quadratic weights",
    }
    log.info("  Scale 0.5x quad:  r = %.4f  (drop %.4f = %.1f%%)", r, full_r - r,
             100 * (full_r - r) / full_r)

    # --- Perturbation 6: Gaussian noise on sensitivity quad ---
    rng = np.random.RandomState(42)
    noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
    noise_results = {}
    for sigma in noise_levels:
        trials = []
        for trial in range(5):
            g = genome.copy()
            for k in sens_quad_indices:
                g[D + k] += rng.randn() * sigma
            r = evaluate_genome(g, measures, quad, targets)
            trials.append(r)
        mean_r = float(np.mean(trials))
        std_r = float(np.std(trials))
        noise_results[str(sigma)] = {
            "mean_r": mean_r, "std_r": std_r,
            "drop": full_r - mean_r,
            "drop_pct": 100 * (full_r - mean_r) / full_r,
        }
        log.info("  Noise σ=%.2f:      r = %.4f ± %.4f  (drop %.1f%%)",
                 sigma, mean_r, std_r, 100 * (full_r - mean_r) / full_r)

    results["perturbations"]["gaussian_noise"] = noise_results

    # --- Perturbation 7: Swap sensitivity with another feature ---
    log.info("  --- Feature swap experiments ---")
    swap_results = {}
    for fi, fname in enumerate(MEASURE_NAMES):
        if fi == si:
            continue
        g = genome.copy()
        # swap linear weights
        g[si], g[fi] = g[fi], g[si]
        # swap quadratic columns (sensitivity <-> fi)
        q = genome[Q_SLICE].copy()
        q_new = q.copy()
        # rebuild interaction matrix, swap rows/cols si and fi
        mat = np.zeros((D, D))
        k = 0
        for i in range(D):
            for j in range(i, D):
                mat[i, j] = q[k]
                mat[j, i] = q[k]
                k += 1
        # swap rows
        mat[[si, fi]] = mat[[fi, si]]
        # swap cols
        mat[:, [si, fi]] = mat[:, [fi, si]]
        # rebuild upper triangle
        k = 0
        for i in range(D):
            for j in range(i, D):
                q_new[k] = mat[i, j]
                k += 1
        g[Q_SLICE] = q_new
        # swap threshold params
        g[D + Q_DIM + si], g[D + Q_DIM + fi] = g[D + Q_DIM + fi], g[D + Q_DIM + si]
        g[D + Q_DIM + D + si], g[D + Q_DIM + D + fi] = g[D + Q_DIM + D + fi], g[D + Q_DIM + D + si]

        r = evaluate_genome(g, measures, quad, targets)
        swap_results[fname] = {"r": r, "drop": full_r - r}
        log.info("  Swap sensitivity <-> %-20s: r = %.4f  (drop %.4f)", fname, r, full_r - r)

    results["perturbations"]["feature_swap"] = swap_results

    log.info("  === Summary ===")
    log.info("  Baseline: %.4f", full_r)
    for name, p in results["perturbations"].items():
        if isinstance(p, dict) and "r" in p:
            log.info("  %-25s  r=%.4f  drop=%.1f%%", name, p["r"], p.get("drop_pct", 0))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Attack 4: Barrier phase diagram (from existing convergence data)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_diagram():
    """Generate barrier phase diagram from existing robustness + main search data."""
    log.info("=== ATTACK: Barrier phase diagram ===")

    # load robustness results
    rob_path = os.path.join(PROJECT_DIR, "data", "robustness_results.json")
    if not os.path.exists(rob_path):
        log.error("No robustness_results.json found. Run robustness.py first.")
        return None

    with open(rob_path) as f:
        rob = json.load(f)

    # load mechanism results for sensitivity importance
    mech_path = os.path.join(PROJECT_DIR, "data", "mechanism_results.json")
    if os.path.exists(mech_path):
        with open(mech_path) as f:
            mech = json.load(f)
    else:
        mech = {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Phase diagram: r vs barrier config ---
    barrier_data = rob.get("barrier_ablation", {})
    if not barrier_data:
        log.error("No barrier ablation data found.")
        return None

    configs = []
    rs = []
    sens_weights = []
    for name in ["none", "natural_only", "relativ_only", "algebrize_only", "all_three"]:
        if name not in barrier_data:
            continue
        entry = barrier_data[name]
        configs.append(name)
        rs.append(entry["analysis"]["full_r"])
        sens_weights.append(entry["analysis"]["importance"].get("sensitivity", 0))

    # Figure 1: r and sensitivity weight by barrier config
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    x = range(len(configs))
    ax1.bar(x, rs, color='#1976d2', alpha=0.8)
    ax1.set_ylabel("Spearman r")
    ax1.set_title("Barrier phase diagram: performance vs constraint")
    for i, r in enumerate(rs):
        ax1.text(i, r + 0.005, f"{r:.3f}", ha='center', fontsize=9)

    ax2.bar(x, sens_weights, color='#d32f2f', alpha=0.8)
    ax2.set_ylabel("Sensitivity importance")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=9)
    for i, w in enumerate(sens_weights):
        ax2.text(i, w + 0.002, f"{w:.3f}", ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "barrier_phase_diagram.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved barrier_phase_diagram.pdf")

    # Figure 2: convergence comparison across barrier configs
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'none': '#4caf50', 'natural_only': '#ff9800', 'relativ_only': '#9c27b0',
              'algebrize_only': '#2196f3', 'all_three': '#d32f2f'}

    for name in ["none", "natural_only", "relativ_only", "algebrize_only", "all_three"]:
        if name not in barrier_data:
            continue
        conv = barrier_data[name].get("convergence", [])
        gens = [c["gen"] for c in conv]
        best_rs = [c.get("best_r", c.get("feas_r", 0)) for c in conv]
        feas_rs = [c.get("feas_r", c.get("best_r", 0)) for c in conv]
        ax.plot(gens, best_rs, 'o-', color=colors.get(name, 'gray'),
                label=name, markersize=4)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Spearman r")
    ax.set_title("Convergence by barrier configuration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "barrier_convergence.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved barrier_convergence.pdf")

    # Figure 3: alt searchers comparison
    alt_data = rob.get("alt_searchers", {})
    if alt_data:
        methods = []
        method_rs = []
        top_features = []

        # add CMA-ES baseline from barrier ablation
        if "all_three" in barrier_data:
            methods.append("CMA-ES\n(barrier-aware)")
            method_rs.append(barrier_data["all_three"]["analysis"]["full_r"])
            top_features.append(barrier_data["all_three"]["analysis"]["top_signal"][:2])

        for name, entry in alt_data.items():
            methods.append(name.replace("_", "\n"))
            method_rs.append(entry["analysis"]["full_r"])
            top_features.append(entry["analysis"]["top_signal"][:2])

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(range(len(methods)), method_rs,
                      color=['#d32f2f', '#1976d2', '#4caf50'][:len(methods)])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel("Spearman r")
        ax.set_title("Searcher independence: same signal across methods")
        for i, (bar, r, top) in enumerate(zip(bars, method_rs, top_features)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"r={r:.3f}\n{top[0][0][:8]}, {top[1][0][:8]}" if len(top) >= 2 else f"r={r:.3f}",
                    ha='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "searcher_comparison.pdf"), dpi=150)
        plt.close(fig)
        log.info("  Saved searcher_comparison.pdf")

    result = {
        "configs": configs,
        "r_values": rs,
        "sensitivity_weights": sens_weights,
        "figures": ["barrier_phase_diagram.pdf", "barrier_convergence.pdf",
                     "searcher_comparison.pdf"],
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]
    results = load_results()

    if exp == "phase_diagram":
        result = run_phase_diagram()
        if result:
            results["phase_diagram"] = result
            save_results(results)
        return

    log.info("Setting up evaluators...")
    setup_data = setup()

    if exp == "cross_only":
        result = run_cross_only(setup_data)
        results["cross_only"] = result
        save_results(results)

    elif exp == "label_random":
        result = run_label_random(setup_data)
        results["label_random"] = result
        save_results(results)

    elif exp == "perturb":
        result = run_perturb(setup_data)
        results["perturb"] = result
        save_results(results)

    else:
        log.error("Unknown experiment: %s", exp)
        sys.exit(1)


if __name__ == "__main__":
    main()
