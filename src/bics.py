"""
Barrier-Induced Cancellation Subspace (BICS) — formal characterization.

Experiments:
  dof_decomp    — DOF budget: per-parameter sensitivity to r vs feasibility
  penalty_sweep — continuous phase transition: cancellation weight vs penalty strength
  all           — both experiments

Usage: python bics.py <experiment>
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
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "bics.log")),
    ],
)
log = logging.getLogger(__name__)

from mechanism import (
    MEASURE_NAMES, D, Q_DIM, GENOME_DIM, W_SLICE, Q_SLICE, A_SLICE, T_SLICE,
    load_data, load_best_genome,
)
from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
from barriers import check_all_barriers_batch, RelativizationChecker
from search import CMAES
from analyze_candidates import feature_importance

RESULT_PATH = os.path.join(PROJECT_DIR, "data", "bics_results.json")


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


def get_sensitivity_indices():
    """All genome indices belonging to the sensitivity cancellation subspace."""
    si = MEASURE_NAMES.index("sensitivity")
    indices = set()
    indices.add(si)  # linear
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                indices.add(D + k)
            k += 1
    indices.add(D + Q_DIM + si)      # threshold activation
    indices.add(D + Q_DIM + D + si)  # threshold position
    return sorted(indices)


def evaluate_barriers(genome, knuth_eval, random_eval, rel_checker, cfg):
    """Evaluate barrier pass/fail for a single genome."""
    mu_knuth = knuth_eval.evaluate_batch(genome[None, :])
    mu_random = random_eval.evaluate_batch(genome[None, :])
    bp, bd = check_all_barriers_batch(
        mu_knuth, mu_random, genome[None, :], rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )
    return bp[0], bd[0]


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: DOF decomposition
# ═══════════════════════════════════════════════════════════════════════════

def dof_decomp(setup_data):
    """Per-parameter finite-difference: dr/dtheta and dfeasibility/dtheta."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data
    genome = load_best_genome()

    log.info("=" * 70)
    log.info("EXPERIMENT 1: DOF DECOMPOSITION")
    log.info("=" * 70)

    # baseline
    mu_base = knuth_eval.evaluate_single(genome)
    r_base, _ = spearmanr(mu_base, targets)
    bp_base, bd_base = evaluate_barriers(genome, knuth_eval, random_eval, rel_checker, cfg)
    log.info("  Baseline: r=%.4f, feasible=%s", r_base, bp_base)

    cancel_idx = set(get_sensitivity_indices())
    signal_idx = set(range(GENOME_DIM)) - cancel_idx
    log.info("  Cancellation subspace: %d params", len(cancel_idx))
    log.info("  Signal subspace: %d params", len(signal_idx))

    eps = 0.01
    dr = np.zeros(GENOME_DIM)
    dfeas = np.zeros(GENOME_DIM)

    # batch evaluate perturbations: +eps and -eps for each parameter
    # do batches of 85 (one per param) for +eps, then 85 for -eps
    perturbed_plus = np.tile(genome, (GENOME_DIM, 1))
    perturbed_minus = np.tile(genome, (GENOME_DIM, 1))
    for i in range(GENOME_DIM):
        perturbed_plus[i, i] += eps
        perturbed_minus[i, i] -= eps

    # evaluate r for all perturbations
    mu_plus = knuth_eval.evaluate_batch(perturbed_plus)
    mu_minus = knuth_eval.evaluate_batch(perturbed_minus)

    for i in range(GENOME_DIM):
        r_plus, _ = spearmanr(mu_plus[i], targets)
        r_minus, _ = spearmanr(mu_minus[i], targets)
        dr[i] = (r_plus - r_minus) / (2 * eps)

    # evaluate feasibility (barrier details) for each perturbation
    # need to check barrier pass/fail change
    log.info("  Computing feasibility gradients (170 barrier evaluations)...")
    for i in range(GENOME_DIM):
        # check if perturbing this param breaks/fixes any barrier
        bp_plus, bd_plus = evaluate_barriers(perturbed_plus[i], knuth_eval, random_eval,
                                              rel_checker, cfg)
        bp_minus, bd_minus = evaluate_barriers(perturbed_minus[i], knuth_eval, random_eval,
                                                rel_checker, cfg)

        # feasibility gradient: count barrier violations
        n_viol_plus = sum(1 for v in bd_plus.values() if not v["passes"])
        n_viol_minus = sum(1 for v in bd_minus.values() if not v["passes"])
        dfeas[i] = (n_viol_minus - n_viol_plus) / (2 * eps)  # positive = increases feasibility

    # aggregate by subspace
    signal_dr = np.abs(dr[list(signal_idx)])
    cancel_dr = np.abs(dr[list(cancel_idx)])
    signal_dfeas = np.abs(dfeas[list(signal_idx)])
    cancel_dfeas = np.abs(dfeas[list(cancel_idx)])

    log.info("")
    log.info("  === DOF DECOMPOSITION ===")
    log.info("  %20s  %12s  %12s  %12s  %12s",
             "Subspace", "mean|dr|", "sum|dr|", "mean|dfeas|", "sum|dfeas|")
    log.info("  %20s  %12.6f  %12.4f  %12.6f  %12.4f",
             "Signal (%d)" % len(signal_idx),
             signal_dr.mean(), signal_dr.sum(),
             signal_dfeas.mean(), signal_dfeas.sum())
    log.info("  %20s  %12.6f  %12.4f  %12.6f  %12.4f",
             "Cancellation (%d)" % len(cancel_idx),
             cancel_dr.mean(), cancel_dr.sum(),
             cancel_dfeas.mean(), cancel_dfeas.sum())

    # ratio
    r_ratio = cancel_dr.mean() / (signal_dr.mean() + 1e-12)
    f_ratio = cancel_dfeas.mean() / (signal_dfeas.mean() + 1e-12)
    log.info("")
    log.info("  Cancel/Signal ratio: dr=%.3f, dfeas=%.3f", r_ratio, f_ratio)
    if r_ratio < 0.5 and f_ratio > 2.0:
        log.info("  [CONFIRMED] Cancellation subspace: LOW r-sensitivity, HIGH feas-sensitivity")
    elif r_ratio < 1.0 and f_ratio > 1.0:
        log.info("  [PARTIAL] Cancellation subspace shows expected asymmetry")

    # per-parameter detail for top contributors
    log.info("")
    log.info("  Top 10 parameters by |dr/dtheta|:")
    order_r = np.argsort(-np.abs(dr))
    for rank, i in enumerate(order_r[:10]):
        subspace = "CANCEL" if i in cancel_idx else "signal"
        param_name = _param_name(i)
        log.info("    %2d. [%s] param %3d %-30s dr=%+.6f  dfeas=%+.4f",
                 rank + 1, subspace, i, param_name, dr[i], dfeas[i])

    log.info("")
    log.info("  Top 10 parameters by |dfeas/dtheta|:")
    order_f = np.argsort(-np.abs(dfeas))
    for rank, i in enumerate(order_f[:10]):
        subspace = "CANCEL" if i in cancel_idx else "signal"
        param_name = _param_name(i)
        log.info("    %2d. [%s] param %3d %-30s dr=%+.6f  dfeas=%+.4f",
                 rank + 1, subspace, i, param_name, dr[i], dfeas[i])

    result = {
        "eps": eps,
        "r_base": float(r_base),
        "feasible_base": bool(bp_base),
        "n_cancel_params": len(cancel_idx),
        "n_signal_params": len(signal_idx),
        "signal": {
            "mean_abs_dr": float(signal_dr.mean()),
            "sum_abs_dr": float(signal_dr.sum()),
            "mean_abs_dfeas": float(signal_dfeas.mean()),
            "sum_abs_dfeas": float(signal_dfeas.sum()),
        },
        "cancellation": {
            "mean_abs_dr": float(cancel_dr.mean()),
            "sum_abs_dr": float(cancel_dr.sum()),
            "mean_abs_dfeas": float(cancel_dfeas.mean()),
            "sum_abs_dfeas": float(cancel_dfeas.sum()),
        },
        "ratio_cancel_signal_dr": float(r_ratio),
        "ratio_cancel_signal_dfeas": float(f_ratio),
        "dr_per_param": dr.tolist(),
        "dfeas_per_param": dfeas.tolist(),
    }

    # generate figure
    _plot_dof(dr, dfeas, cancel_idx, signal_idx)

    return result


def _param_name(idx):
    """Human-readable name for genome parameter index."""
    if idx < D:
        return f"W[{MEASURE_NAMES[idx]}]"
    elif idx < D + Q_DIM:
        k = idx - D
        i, j = 0, 0
        count = 0
        for ii in range(D):
            for jj in range(ii, D):
                if count == k:
                    i, j = ii, jj
                count += 1
        name_i = MEASURE_NAMES[i][:8]
        name_j = MEASURE_NAMES[j][:8]
        return f"Q[{name_i} x {name_j}]"
    elif idx < D + Q_DIM + D:
        fi = idx - D - Q_DIM
        return f"A[{MEASURE_NAMES[fi]}]"
    else:
        fi = idx - D - Q_DIM - D
        return f"T[{MEASURE_NAMES[fi]}]"


def _plot_dof(dr, dfeas, cancel_idx, signal_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: scatter dr vs dfeas, colored by subspace
    ax = axes[0]
    for i in range(GENOME_DIM):
        color = '#d32f2f' if i in cancel_idx else '#1976d2'
        alpha = 0.7 if i in cancel_idx else 0.3
        ax.scatter(abs(dr[i]), abs(dfeas[i]), c=color, alpha=alpha, s=20, edgecolors='none')
    ax.set_xlabel("|dr/dtheta| (correlation sensitivity)")
    ax.set_ylabel("|dfeas/dtheta| (feasibility sensitivity)")
    ax.set_title("A. DOF decomposition: each parameter")
    # legend
    ax.scatter([], [], c='#d32f2f', label='Cancellation (sensitivity)', s=30)
    ax.scatter([], [], c='#1976d2', label='Signal (other features)', s=30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: bar chart of mean |dr| and mean |dfeas| by subspace
    ax = axes[1]
    signal_dr = np.abs(dr[list(signal_idx)])
    cancel_dr = np.abs(dr[list(cancel_idx)])
    signal_dfeas = np.abs(dfeas[list(signal_idx)])
    cancel_dfeas = np.abs(dfeas[list(cancel_idx)])

    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width / 2, [signal_dr.mean(), cancel_dr.mean()], width,
                   label='mean |dr/dtheta|', color='#1976d2')
    bars2 = ax.bar(x + width / 2, [signal_dfeas.mean(), cancel_dfeas.mean()], width,
                   label='mean |dfeas/dtheta|', color='#d32f2f')
    ax.set_xticks(x)
    ax.set_xticklabels(['Signal\n(73 params)', 'Cancellation\n(12 params)'])
    ax.set_title("B. Subspace sensitivity comparison")
    ax.legend(fontsize=8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.5f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)

    # Panel C: parameter index heatmap
    ax = axes[2]
    dr_norm = np.abs(dr) / (np.abs(dr).max() + 1e-12)
    dfeas_norm = np.abs(dfeas) / (np.abs(dfeas).max() + 1e-12)
    data = np.stack([dr_norm, dfeas_norm])
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                   extent=[0, GENOME_DIM, 1.5, -0.5])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['|dr/dtheta|', '|dfeas/dtheta|'])
    ax.set_xlabel('Parameter index')
    ax.set_title('C. Sensitivity heatmap')
    # mark cancellation region
    for idx in sorted(cancel_idx):
        ax.axvline(idx, color='red', alpha=0.1, linewidth=0.5)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized sensitivity')

    fig.suptitle("BICS: Barrier-Induced Cancellation Subspace characterization", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bics_dof_decomp.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved bics_dof_decomp.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Penalty sweep (continuous phase transition)
# ═══════════════════════════════════════════════════════════════════════════

def penalty_sweep(setup_data):
    """Sweep barrier penalty from 0 to 50, track cancellation weight emergence."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    log.info("=" * 70)
    log.info("EXPERIMENT 2: PENALTY SWEEP")
    log.info("=" * 70)

    cancel_idx = set(get_sensitivity_indices())
    penalties = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    results = {}

    for pen in penalties:
        log.info("")
        log.info("  === Penalty = %.1f ===", pen)
        t0 = time.time()

        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=100, seed=42)

        best_r = 0.0
        best_genome = None
        best_feas_r = 0.0
        best_feas_genome = None
        per_gen = []
        stall = 0
        prev_best = 0.0

        for gen in range(80):
            candidates = es.ask()
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
            fitness = -corr + pen * n_violations + l2_lambda * l2_norm
            es.tell(candidates, fitness)

            if es.should_restart():
                es.restart()

            idx_best = corr.argmax()
            if corr[idx_best] > best_r:
                best_r = float(corr[idx_best])
                best_genome = candidates[idx_best].copy()

            feas_corrs = corr[barrier_pass.astype(bool)]
            if len(feas_corrs) > 0 and feas_corrs.max() > best_feas_r:
                best_feas_r = float(feas_corrs.max())
                best_feas_genome = candidates[barrier_pass.astype(bool)][feas_corrs.argmax()].copy()

            # track per-gen: sensitivity weight fraction of best genome
            imp = feature_importance(best_genome, D)
            si = MEASURE_NAMES.index("sensitivity")
            sens_frac = float(imp[si])

            per_gen.append({
                "gen": gen + 1,
                "best_r": float(corr.max()),
                "best_feas_r": float(feas_corrs.max()) if len(feas_corrs) > 0 else 0.0,
                "n_feasible": int(barrier_pass.sum()),
                "sensitivity_importance": sens_frac,
            })

            if best_r > prev_best + 1e-5:
                prev_best = best_r
                stall = 0
            else:
                stall += 1
            if stall >= 60:
                log.info("    Early stop at gen %d", gen + 1)
                break

        elapsed = time.time() - t0

        # final analysis of best genome
        final_imp = feature_importance(best_genome, D)
        sens_imp = float(final_imp[si])

        # cancellation subspace weight fraction
        cancel_weight = float(np.abs(best_genome[list(cancel_idx)]).sum())
        total_weight = float(np.abs(best_genome).sum())
        cancel_frac = cancel_weight / (total_weight + 1e-12)

        log.info("    best_r=%.4f  feas_r=%.4f  feasible=%d/100",
                 best_r, best_feas_r, per_gen[-1]["n_feasible"])
        log.info("    sensitivity importance=%.3f  cancel weight frac=%.3f",
                 sens_imp, cancel_frac)
        log.info("    (%.1f min)", elapsed / 60)

        results[str(pen)] = {
            "penalty": pen,
            "best_r": best_r,
            "best_feas_r": best_feas_r,
            "n_feasible_final": per_gen[-1]["n_feasible"],
            "sensitivity_importance": sens_imp,
            "cancel_weight_fraction": cancel_frac,
            "per_gen": per_gen,
            "elapsed_s": elapsed,
        }

    # generate figure
    _plot_penalty_sweep(results, penalties)

    return results


def _plot_penalty_sweep(results, penalties):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    pens = [p for p in penalties if str(p) in results]
    rs = [results[str(p)]["best_r"] for p in pens]
    feas_rs = [results[str(p)]["best_feas_r"] for p in pens]
    sens_imps = [results[str(p)]["sensitivity_importance"] for p in pens]
    cancel_fracs = [results[str(p)]["cancel_weight_fraction"] for p in pens]
    n_feas = [results[str(p)]["n_feasible_final"] for p in pens]

    # Panel A: r vs penalty
    ax = axes[0, 0]
    ax.plot(pens, rs, 'o-', color='#1976d2', label='Best r (any)', markersize=6)
    ax.plot(pens, feas_rs, 's-', color='#d32f2f', label='Best r (feasible)', markersize=6)
    ax.set_xlabel("Barrier penalty")
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Performance vs constraint strength")
    ax.legend(fontsize=8)
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)

    # Panel B: sensitivity importance vs penalty
    ax = axes[0, 1]
    ax.plot(pens, sens_imps, 'D-', color='#d32f2f', markersize=6)
    ax.set_xlabel("Barrier penalty")
    ax.set_ylabel("Sensitivity importance fraction")
    ax.set_title("B. Cancellation module emergence")
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)
    for i, (p, s) in enumerate(zip(pens, sens_imps)):
        ax.annotate(f'{s:.2f}', (p, s), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7)

    # Panel C: cancellation weight fraction vs penalty
    ax = axes[1, 0]
    ax.plot(pens, cancel_fracs, '^-', color='#9c27b0', markersize=6)
    ax.set_xlabel("Barrier penalty")
    ax.set_ylabel("Cancellation subspace weight fraction")
    ax.set_title("C. DOF consumption by BICS")
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)

    # Panel D: feasibility rate vs penalty
    ax = axes[1, 1]
    ax.plot(pens, n_feas, 'o-', color='#4caf50', markersize=6)
    ax.set_xlabel("Barrier penalty")
    ax.set_ylabel("Feasible solutions / 100")
    ax.set_title("D. Feasibility rate")
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Penalty sweep: continuous emergence of barrier-induced cancellation",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bics_penalty_sweep.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved bics_penalty_sweep.pdf")

    # convergence trajectories for selected penalties
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {0.0: '#4caf50', 0.5: '#ff9800', 2.0: '#2196f3',
              10.0: '#d32f2f', 50.0: '#9c27b0'}
    for pen in [0.0, 0.5, 2.0, 10.0, 50.0]:
        key = str(pen)
        if key not in results:
            continue
        gens = [g["gen"] for g in results[key]["per_gen"]]
        sens = [g["sensitivity_importance"] for g in results[key]["per_gen"]]
        ax.plot(gens, sens, '-', color=colors.get(pen, 'gray'),
                label=f'penalty={pen}', linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Sensitivity importance fraction")
    ax.set_title("Cancellation module emergence over search time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bics_emergence.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved bics_emergence.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]

    log.info("Setting up evaluators...")
    setup_data = setup()
    results = load_results()

    if exp in ("dof_decomp", "all"):
        result = dof_decomp(setup_data)
        results["dof_decomp"] = result
        save_results(results)

    if exp in ("penalty_sweep", "all"):
        result = penalty_sweep(setup_data)
        results["penalty_sweep"] = result
        save_results(results)

    log.info("All done.")


if __name__ == "__main__":
    main()
