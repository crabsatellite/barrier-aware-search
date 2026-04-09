"""
Barrier-aware search dynamics — compliance basin discovery.

Experiments:
  run <max_gen>                — CMA-ES with per-gen diagnostics (checkpoint/resume)
  warm_start                   — compare warm-start vs cold-start
  truncation                   — progressive truncation on seed 42 (from checkpoint)
  truncation_seed <seed>       — truncation on a saved seed genome
  multiseed [max] [s...]       — multi-seed dynamics with genome saving
  endpoint <seed> [seed...]    — full basin diagnostics for saved genome(s)
  interpolate <sa> <sb> [n]    — linear interpolation connectivity test
  basin_volume <seed> [n]      — random perturbation volume probe
  injection <seed>             — sensitivity injection into a genome
  local_cont <seed> [n] [g]    — perturb + short refinement (basin vs spike)
  targeted <n> [gen] [lam]     — overhead-penalized search (Type C recovery)
  two_stage <n> [s1] [s2]      — signal-first then barrier-aware
  hitrate <n> [gen]            — many-seed regime hit-rate statistics
  irreversibility [s1 s2 ...]  — Type B→C return test (one-way degradation?)
  ablation [seed]              — one-dim CANCEL_IDX ablation on Type C genome
  cov_mech [s1 s2 ...]         — covariance mechanism diagnostic (frozen/reset/projection)
  opt_compare [gen] [s1 s2 ..] — full vs sep vs isotropic CMA-ES drift comparison
  plot                         — generate figures from collected data

Usage:
  python dynamics.py run 8000
  python dynamics.py truncation
  python dynamics.py truncation_seed 137
  python dynamics.py multiseed 8000 42 137 256
  python dynamics.py endpoint 42 137
  python dynamics.py interpolate 42 137 21
  python dynamics.py local_cont 512 10 100
  python dynamics.py targeted 10 4000 0.05
  python dynamics.py two_stage 10 2000 4000
  python dynamics.py hitrate 30 4000
  python dynamics.py plot
"""

import json
import logging
import os
import sys
import time
import pickle

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
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "dynamics.log")),
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

RESULT_PATH = os.path.join(PROJECT_DIR, "data", "dynamics_results.json")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "data", "dynamics_checkpoint.pkl")


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {"trajectory": [], "dof_snapshots": []}


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


def get_cancel_indices():
    si = MEASURE_NAMES.index("sensitivity")
    indices = set()
    indices.add(si)
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                indices.add(D + k)
            k += 1
    indices.add(D + Q_DIM + si)
    indices.add(D + Q_DIM + D + si)
    return sorted(indices)


CANCEL_IDX = get_cancel_indices()
SIGNAL_IDX = sorted(set(range(GENOME_DIM)) - set(CANCEL_IDX))


def compute_diagnostics(genome, measures, quad, targets,
                        knuth_eval, random_eval, rel_checker, cfg):
    """Lightweight per-generation diagnostics."""
    si = MEASURE_NAMES.index("sensitivity")

    # feature importance
    imp = feature_importance(genome, D)
    sens_imp = float(imp[si])

    # weight norms by subspace
    cancel_l1 = float(np.abs(genome[CANCEL_IDX]).sum())
    signal_l1 = float(np.abs(genome[SIGNAL_IDX]).sum())
    total_l1 = cancel_l1 + signal_l1
    cancel_frac = cancel_l1 / (total_l1 + 1e-12)

    # cancellation residual: evaluate sensitivity module alone
    g_sens_only = np.zeros_like(genome)
    for idx in CANCEL_IDX:
        g_sens_only[idx] = genome[idx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MeasureEvaluator(measures, quad, device)
    mu_sens = ev.evaluate_single(g_sens_only)
    cancel_mean = float(np.mean(mu_sens))
    cancel_std = float(np.std(mu_sens))
    cancel_abs_mean = float(np.mean(np.abs(mu_sens)))

    # barrier margin: evaluate full genome
    mu_knuth = knuth_eval.evaluate_batch(genome[None, :])
    mu_random = random_eval.evaluate_batch(genome[None, :])
    bp, bd = check_all_barriers_batch(
        mu_knuth, mu_random, genome[None, :], rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )

    barrier_margins = {}
    for bname, bdetail in bd[0].items():
        barrier_margins[bname] = {
            "passes": bdetail["passes"],
            "value": float(bdetail.get("value", 0)),
        }

    # sensitivity opposing structure: linear vs quadratic contribution
    g_lin = np.zeros_like(genome)
    g_lin[si] = genome[si]
    mu_lin = ev.evaluate_single(g_lin)
    lin_mean = float(np.mean(mu_lin))

    g_quad = np.zeros_like(genome)
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                g_quad[D + k] = genome[D + k]
            k += 1
    mu_quad = ev.evaluate_single(g_quad)
    quad_mean = float(np.mean(mu_quad))

    return {
        "sensitivity_importance": sens_imp,
        "cancel_weight_frac": cancel_frac,
        "cancel_l1": cancel_l1,
        "signal_l1": signal_l1,
        "cancel_residual_mean": cancel_mean,
        "cancel_residual_std": cancel_std,
        "cancel_residual_abs_mean": cancel_abs_mean,
        "sens_linear_mean": lin_mean,
        "sens_quadratic_mean": quad_mean,
        "opposition_ratio": abs(lin_mean / (quad_mean + 1e-12)),
        "barrier_margins": barrier_margins,
    }


def compute_dof_snapshot(genome, measures, quad, targets,
                         knuth_eval, random_eval, rel_checker, cfg):
    """Full DOF decomposition at a checkpoint (expensive: 170 barrier evaluations)."""
    eps = 0.01
    dr = np.zeros(GENOME_DIM)
    dfeas = np.zeros(GENOME_DIM)

    perturbed_plus = np.tile(genome, (GENOME_DIM, 1))
    perturbed_minus = np.tile(genome, (GENOME_DIM, 1))
    for i in range(GENOME_DIM):
        perturbed_plus[i, i] += eps
        perturbed_minus[i, i] -= eps

    mu_plus = knuth_eval.evaluate_batch(perturbed_plus)
    mu_minus = knuth_eval.evaluate_batch(perturbed_minus)

    for i in range(GENOME_DIM):
        r_plus, _ = spearmanr(mu_plus[i], targets)
        r_minus, _ = spearmanr(mu_minus[i], targets)
        dr[i] = (r_plus - r_minus) / (2 * eps)

    for i in range(GENOME_DIM):
        bp_p, bd_p = check_all_barriers_batch(
            knuth_eval.evaluate_batch(perturbed_plus[i:i+1]),
            random_eval.evaluate_batch(perturbed_plus[i:i+1]),
            perturbed_plus[i:i+1], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        bp_m, bd_m = check_all_barriers_batch(
            knuth_eval.evaluate_batch(perturbed_minus[i:i+1]),
            random_eval.evaluate_batch(perturbed_minus[i:i+1]),
            perturbed_minus[i:i+1], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_viol_p = sum(1 for v in bd_p[0].values() if not v["passes"])
        n_viol_m = sum(1 for v in bd_m[0].values() if not v["passes"])
        dfeas[i] = (n_viol_m - n_viol_p) / (2 * eps)

    cancel_dr = np.abs(dr[CANCEL_IDX])
    signal_dr = np.abs(dr[SIGNAL_IDX])
    cancel_dfeas = np.abs(dfeas[CANCEL_IDX])
    signal_dfeas = np.abs(dfeas[SIGNAL_IDX])

    return {
        "signal_mean_dr": float(signal_dr.mean()),
        "cancel_mean_dr": float(cancel_dr.mean()),
        "signal_mean_dfeas": float(signal_dfeas.mean()),
        "cancel_mean_dfeas": float(cancel_dfeas.mean()),
        "dr_ratio": float(cancel_dr.mean() / (signal_dr.mean() + 1e-12)),
        "dfeas_ratio": float(cancel_dfeas.mean() / (signal_dfeas.mean() + 1e-12)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main search with dynamics tracking
# ═══════════════════════════════════════════════════════════════════════════

def run_dynamics(setup_data, max_gen):
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    results = load_results()
    existing_gens = {t["gen"] for t in results["trajectory"]}
    existing_dof_gens = {s["gen"] for s in results["dof_snapshots"]}

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100

    # resume or start fresh
    start_gen = 0
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        es = ckpt["es"]
        best_genome = ckpt["best_genome"]
        best_r = ckpt["best_r"]
        start_gen = ckpt["gen"]
        log.info("Resumed from gen %d (best_r=%.4f)", start_gen, best_r)
    else:
        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=42)
        best_genome = None
        best_r = 0.0
        log.info("Starting fresh search")

    if start_gen >= max_gen:
        log.info("Already at gen %d >= max_gen %d", start_gen, max_gen)
        return results

    log.info("Running gen %d -> %d", start_gen, max_gen)

    # diagnostic intervals
    LIGHT_INTERVAL = 10    # lightweight diagnostics every 10 gens
    DOF_INTERVAL = 200     # full DOF decomposition every 200 gens
    DOF_EARLY = {50, 100}  # additional early DOF snapshots

    stall = 0
    prev_best = best_r

    for gen in range(start_gen, max_gen):
        candidates = es.ask()
        mu_knuth = knuth_eval.evaluate_batch(candidates)
        mu_random = random_eval.evaluate_batch(candidates)
        corr = spearman_batch_gpu(mu_knuth, target_ranks)

        barrier_pass, barrier_details = check_all_barriers_batch(
            mu_knuth, mu_random, candidates, rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_violations = np.zeros(pop_size, dtype=np.int32)
        for i in range(pop_size):
            if not barrier_pass[i]:
                n_violations[i] = sum(
                    1 for v in barrier_details[i].values() if not v["passes"])

        l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
        fitness = -corr + penalty * n_violations + l2_lambda * l2_norm
        es.tell(candidates, fitness)

        if es.should_restart():
            es.restart()
            log.info("  *** RESTART #%d at gen %d (sigma collapsed) ***",
                     es.n_restarts, gen + 1)

        idx_best = corr.argmax()
        if corr[idx_best] > best_r:
            best_r = float(corr[idx_best])
            best_genome = candidates[idx_best].copy()

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall = 0
        else:
            stall += 1

        gen_num = gen + 1
        feas_corrs = corr[barrier_pass.astype(bool)]
        feas_best = float(feas_corrs.max()) if len(feas_corrs) > 0 else 0.0
        n_feas = int(barrier_pass.sum())

        # lightweight diagnostics
        if gen_num % LIGHT_INTERVAL == 0 or gen_num in DOF_EARLY and gen_num not in existing_gens:
            diag = compute_diagnostics(best_genome, measures, quad, targets,
                                       knuth_eval, random_eval, rel_checker, cfg)
            entry = {
                "gen": gen_num,
                "best_r": best_r,
                "feas_r": feas_best,
                "n_feasible": n_feas,
                "sigma": float(es.sigma),
                **diag,
            }
            results["trajectory"].append(entry)

            if gen_num % 100 == 0:
                log.info("  gen %4d | r=%.4f feas=%.4f | feas=%d | sens_imp=%.3f | "
                         "cancel_frac=%.3f | cancel_mean=%.4f | opp_ratio=%.2f",
                         gen_num, best_r, feas_best, n_feas,
                         diag["sensitivity_importance"],
                         diag["cancel_weight_frac"],
                         diag["cancel_residual_mean"],
                         diag["opposition_ratio"])

        # full DOF decomposition at major checkpoints
        if (gen_num % DOF_INTERVAL == 0 or gen_num in DOF_EARLY) and gen_num not in existing_dof_gens:
            log.info("  DOF snapshot at gen %d...", gen_num)
            dof = compute_dof_snapshot(best_genome, measures, quad, targets,
                                       knuth_eval, random_eval, rel_checker, cfg)
            dof["gen"] = gen_num
            dof["best_r"] = best_r
            results["dof_snapshots"].append(dof)
            log.info("    dr_ratio=%.3f  dfeas_ratio=%.3f",
                     dof["dr_ratio"], dof["dfeas_ratio"])

        # periodic save
        if gen_num % 100 == 0:
            save_results(results)
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({
                    "es": es,
                    "best_genome": best_genome,
                    "best_r": best_r,
                    "gen": gen_num,
                }, f)

    # final save
    save_results(results)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({
            "es": es,
            "best_genome": best_genome,
            "best_r": best_r,
            "gen": max_gen,
        }, f)
    log.info("Saved checkpoint at gen %d (best_r=%.4f)", max_gen, best_r)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Warm-start vs cold-start comparison
# ═══════════════════════════════════════════════════════════════════════════

def warm_start_experiment(setup_data):
    """Compare BICS formation speed: warm-start from feasible solution vs cold-start."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    log.info("=" * 70)
    log.info("WARM-START vs COLD-START EXPERIMENT")
    log.info("=" * 70)

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100
    max_gen = 200

    # Get a warm-start genome: 80-gen best from penalty sweep
    # Run a quick 80-gen search to get a feasible starting point
    log.info("  Phase 1: generating warm-start genome (80 gen search)...")
    es_warm_init = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                         pop_size=pop_size, seed=42)
    warm_genome = None
    warm_r = 0.0
    for gen in range(80):
        candidates = es_warm_init.ask()
        mu_k = knuth_eval.evaluate_batch(candidates)
        mu_r = random_eval.evaluate_batch(candidates)
        corr = spearman_batch_gpu(mu_k, target_ranks)
        bp, bd = check_all_barriers_batch(
            mu_k, mu_r, candidates, rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        nv = np.zeros(pop_size, dtype=np.int32)
        for i in range(pop_size):
            if not bp[i]:
                nv[i] = sum(1 for v in bd[i].values() if not v["passes"])
        l2 = np.sqrt((candidates ** 2).sum(axis=1))
        fit = -corr + penalty * nv + l2_lambda * l2
        es_warm_init.tell(candidates, fit)
        if es_warm_init.should_restart():
            es_warm_init.restart()
        idx = corr.argmax()
        if corr[idx] > warm_r:
            warm_r = float(corr[idx])
            warm_genome = candidates[idx].copy()

    log.info("  Warm-start genome: r=%.4f", warm_r)

    # Now run two experiments: warm-start and cold-start
    conditions = {
        "cold_start": {"init_mean": np.zeros(GENOME_DIM), "seed": 99},
        "warm_start": {"init_mean": warm_genome, "seed": 99},
    }

    all_results = {}

    for name, cond in conditions.items():
        log.info("")
        log.info("  === %s (%d gens) ===", name, max_gen)
        t0 = time.time()

        es = CMAES(GENOME_DIM, sigma0=0.5 if name == "warm_start" else 1.0,
                   sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=cond["seed"])
        # set initial mean for warm start
        if name == "warm_start":
            es.mean = cond["init_mean"].copy()

        best_r = 0.0
        best_genome = None
        trajectory = []

        for gen in range(max_gen):
            candidates = es.ask()
            mu_k = knuth_eval.evaluate_batch(candidates)
            mu_r = random_eval.evaluate_batch(candidates)
            corr = spearman_batch_gpu(mu_k, target_ranks)
            bp, bd = check_all_barriers_batch(
                mu_k, mu_r, candidates, rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )
            nv = np.zeros(pop_size, dtype=np.int32)
            for i in range(pop_size):
                if not bp[i]:
                    nv[i] = sum(1 for v in bd[i].values() if not v["passes"])
            l2 = np.sqrt((candidates ** 2).sum(axis=1))
            fit = -corr + penalty * nv + l2_lambda * l2
            es.tell(candidates, fit)
            if es.should_restart():
                es.restart()

            idx = corr.argmax()
            if corr[idx] > best_r:
                best_r = float(corr[idx])
                best_genome = candidates[idx].copy()

            if (gen + 1) % 10 == 0:
                imp = feature_importance(best_genome, D)
                si = MEASURE_NAMES.index("sensitivity")
                cancel_l1 = float(np.abs(best_genome[CANCEL_IDX]).sum())
                total_l1 = float(np.abs(best_genome).sum())

                trajectory.append({
                    "gen": gen + 1,
                    "best_r": best_r,
                    "n_feasible": int(bp.sum()),
                    "sensitivity_importance": float(imp[si]),
                    "cancel_weight_frac": cancel_l1 / (total_l1 + 1e-12),
                })

            if (gen + 1) % 50 == 0:
                log.info("    gen %3d | r=%.4f | feas=%d | sens=%.3f",
                         gen + 1, best_r, int(bp.sum()),
                         trajectory[-1]["sensitivity_importance"])

        elapsed = time.time() - t0
        all_results[name] = {
            "trajectory": trajectory,
            "final_r": best_r,
            "final_sensitivity": trajectory[-1]["sensitivity_importance"],
            "final_cancel_frac": trajectory[-1]["cancel_weight_frac"],
            "elapsed_s": elapsed,
        }
        log.info("  %s: final r=%.4f, sens=%.3f, cancel_frac=%.3f  (%.1f min)",
                 name, best_r, trajectory[-1]["sensitivity_importance"],
                 trajectory[-1]["cancel_weight_frac"], elapsed / 60)

    # generate comparison figure
    _plot_warm_cold(all_results)

    return all_results


def _plot_warm_cold(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, color in [("cold_start", "#1976d2"), ("warm_start", "#d32f2f")]:
        traj = results[name]["trajectory"]
        gens = [t["gen"] for t in traj]
        rs = [t["best_r"] for t in traj]
        sens = [t["sensitivity_importance"] for t in traj]
        cancel = [t["cancel_weight_frac"] for t in traj]

        axes[0].plot(gens, rs, '-', color=color, label=name.replace("_", " "), linewidth=1.5)
        axes[1].plot(gens, sens, '-', color=color, label=name.replace("_", " "), linewidth=1.5)
        axes[2].plot(gens, cancel, '-', color=color, label=name.replace("_", " "), linewidth=1.5)

    axes[0].set_ylabel("Best Spearman r")
    axes[0].set_title("A. Convergence")
    axes[1].set_ylabel("Sensitivity importance")
    axes[1].set_title("B. BICS formation (importance)")
    axes[2].set_ylabel("Cancellation weight fraction")
    axes[2].set_title("C. BICS formation (weight)")

    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Warm-start vs cold-start: BICS formation dynamics", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bics_warm_cold.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved bics_warm_cold.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Plot dynamics from collected data
# ═══════════════════════════════════════════════════════════════════════════

def plot_dynamics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results()
    if not results["trajectory"]:
        log.error("No trajectory data. Run `python dynamics.py run <max_gen>` first.")
        return

    traj = sorted(results["trajectory"], key=lambda t: t["gen"])
    fig_dir = os.path.join(PROJECT_DIR, "figures")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    gens = [t["gen"] for t in traj]

    # A: best r with signal ceiling
    ax = axes[0, 0]
    ax.plot(gens, [t["best_r"] for t in traj], '-', color='#1976d2', linewidth=1)
    ax.axhline(0.5932, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.annotate('signal ceiling (no sensitivity)', xy=(max(gens)*0.55, 0.5935),
                fontsize=7, color='gray', ha='center')
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Convergence")

    # B: sensitivity importance
    ax = axes[0, 1]
    ax.plot(gens, [t["sensitivity_importance"] for t in traj], '-', color='#d32f2f', linewidth=1)
    ax.set_ylabel("Sensitivity importance")
    ax.set_title("B. Cancellation module growth")

    # C: cancel weight fraction
    ax = axes[0, 2]
    ax.plot(gens, [t["cancel_weight_frac"] for t in traj], '-', color='#9c27b0', linewidth=1)
    ax.set_ylabel("Cancel weight / total weight")
    ax.set_title("C. DOF consumption")

    # D: cancellation residual — clip y-axis to avoid initial transient dominating
    ax = axes[1, 0]
    residuals = [t["cancel_residual_mean"] for t in traj]
    ax.plot(gens, residuals, '-', color='#4caf50', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel("Mean cancel output")
    ax.set_title("D. Cancellation residual")
    ax.set_xlabel("Generation")
    # clip y-axis: exclude gen<300 outliers
    stable = [r for g, r in zip(gens, residuals) if g >= 300]
    if stable:
        ymin = min(stable) - 0.5
        ymax = max(stable) + 0.5
        ax.set_ylim(ymin, ymax)

    # E: sigma trajectory (detect restarts)
    ax = axes[1, 1]
    sigmas = [t["sigma"] for t in traj]
    ax.semilogy(gens, sigmas, '-', color='#ff9800', linewidth=1)
    ax.set_ylabel("Step-size (log)")
    ax.set_title("E. CMA-ES sigma (restarts visible)")
    ax.set_xlabel("Generation")

    # detect restarts: sigma jumps by >10x
    restart_gens = []
    for i in range(1, len(sigmas)):
        if sigmas[i] > sigmas[i-1] * 10:
            restart_gens.append(gens[i])

    # add restart markers to ALL panels with labels
    for i, ax_flat in enumerate(axes.flat):
        for j, rg in enumerate(restart_gens):
            ax_flat.axvline(rg, color='red', linestyle=':', alpha=0.5, linewidth=1)
            if i == 1:  # label only on Panel B
                ax_flat.annotate(f'R{j+1}', xy=(rg, ax_flat.get_ylim()[1]*0.95),
                                 fontsize=7, color='red', ha='center', fontweight='bold')

    # F: DOF decomposition snapshots (if available)
    ax = axes[1, 2]
    dof = sorted(results.get("dof_snapshots", []), key=lambda s: s["gen"])
    if dof:
        dof_gens = [s["gen"] for s in dof]
        dr_ratios = [s["dr_ratio"] for s in dof]
        dfeas_ratios = [s["dfeas_ratio"] for s in dof]
        ax.plot(dof_gens, dr_ratios, 'o-', color='#1976d2', label='cancel/signal |dr|', markersize=5)
        ax.plot(dof_gens, dfeas_ratios, 's-', color='#d32f2f', label='cancel/signal |dfeas|', markersize=5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='parity')
        ax.legend(fontsize=8)
    ax.set_ylabel("Cancel/Signal ratio")
    ax.set_title("F. DOF sensitivity evolution")
    ax.set_xlabel("Generation")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("BICS formation dynamics: time-resolved characterization", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bics_dynamics.pdf"), dpi=150)
    plt.close(fig)
    log.info("Saved bics_dynamics.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Progressive truncation — decompose BICS into core vs excess shell
# ═══════════════════════════════════════════════════════════════════════════

def progressive_truncation(setup_data):
    """Scale sensitivity parameters from 0% to 100% and measure r + feasibility.

    Tests whether 69.7% sensitivity importance is a necessary feasibility core
    or contains an optimizer-accumulated excess shell.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    genome = ckpt["best_genome"]
    log.info("Loaded genome from gen %d (best_r=%.4f)", ckpt["gen"], ckpt["best_r"])
    log.info("CANCEL_IDX: %d params, SIGNAL_IDX: %d params", len(CANCEL_IDX), len(SIGNAL_IDX))

    scales = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    results = []

    for scale in scales:
        g = genome.copy()
        g[CANCEL_IDX] *= scale

        # evaluate r
        mu_knuth = knuth_eval.evaluate_batch(g[None, :])
        r = spearman_batch_gpu(mu_knuth, target_ranks)[0]

        # evaluate feasibility
        mu_random = random_eval.evaluate_batch(g[None, :])
        bp, bd = check_all_barriers_batch(
            mu_knuth, mu_random, g[None, :], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_violations = sum(1 for v in bd[0].values() if not v["passes"])
        feasible = bool(bp[0])

        # natural-proof density (the key metric)
        median_k = np.median(np.abs(mu_knuth[0]))
        np_density = float(np.mean(np.abs(mu_random[0]) >= median_k))

        # barrier margins
        margins = {}
        for bname, bdetail in bd[0].items():
            margins[bname] = {
                "passes": bdetail["passes"],
                "density": float(bdetail.get("density", bdetail.get("value", 0))),
            }

        entry = {
            "scale": scale,
            "r": float(r),
            "feasible": feasible,
            "n_violations": n_violations,
            "np_density": np_density,
            "margins": margins,
        }
        results.append(entry)
        log.info("  scale=%.2f | r=%.4f | feas=%s | density=%.4f",
                 scale, r, feasible, np_density)

    # save results
    out_path = os.path.join(PROJECT_DIR, "data", "truncation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved truncation results to %s", out_path)

    # generate figure
    _plot_truncation(results)
    return results


def _plot_truncation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    scales = [r["scale"] for r in results]
    rs = [r["r"] for r in results]
    densities = [r.get("np_density", r["n_violations"]) for r in results]

    # A: r vs scale
    ax = axes[0]
    ax.plot(scales, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=6)
    ax.axhline(results[-1]["r"], color='gray', linestyle='--', alpha=0.3)
    ax.axhline(results[0]["r"], color='#d32f2f', linestyle='--', alpha=0.3)
    ax.annotate(f'full: r={results[-1]["r"]:.4f}', xy=(0.6, results[-1]["r"]),
                fontsize=8, color='gray')
    ax.annotate(f'no sens: r={results[0]["r"]:.4f}', xy=(0.02, results[0]["r"]+0.0002),
                fontsize=8, color='#d32f2f')
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Prediction impact (phase transition)")

    # B: natural-proof density vs scale — the key panel
    ax = axes[1]
    ax.plot(scales, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=6)
    np_thresh = 0.04  # 1/n_vars^poly_deg = 1/25
    ax.axhline(np_thresh, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.annotate('threshold (1/n²)', xy=(0.5, np_thresh + 0.02),
                fontsize=8, color='green')
    # shade feasible region
    ax.fill_between(scales, 0, np_thresh, alpha=0.08, color='green')
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Natural-proof density")
    ax.set_title("B. Barrier compliance (density)")
    ax.set_ylim(-0.02, 1.05)

    # C: dual-axis — r and density overlaid
    ax = axes[2]
    ax2 = ax.twinx()
    l1 = ax.plot(scales, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=5, label='Spearman r')
    l2 = ax2.plot(scales, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=5, label='NP density')
    ax2.axhline(np_thresh, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Spearman r", color='#1976d2')
    ax2.set_ylabel("NP density", color='#d32f2f')
    ax.set_title("C. Prediction-compliance tradeoff")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='center left')

    for a in [axes[0], axes[1]]:
        a.grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Progressive truncation: natural-proof compliance mechanism", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", "bics_truncation.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved bics_truncation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Per-seed truncation — compare feasibility islands across seeds
# ═══════════════════════════════════════════════════════════════════════════

def truncation_seed(setup_data, seed_id):
    """Run progressive truncation on a saved seed genome.

    Loads genome from data/genome_seed_{seed_id}.npy and runs the same
    truncation analysis as progressive_truncation(), saving results to
    data/truncation_seed_{seed_id}.json.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    genome_path = os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy")
    if not os.path.exists(genome_path):
        log.error("Genome not found: %s — run multiseed first", genome_path)
        sys.exit(1)

    genome = np.load(genome_path)
    log.info("Loaded seed %d genome from %s (%d params)", seed_id, genome_path, len(genome))
    log.info("CANCEL_IDX: %d params, SIGNAL_IDX: %d params", len(CANCEL_IDX), len(SIGNAL_IDX))

    # baseline evaluation
    mu_k_base = knuth_eval.evaluate_batch(genome[None, :])
    r_base = spearman_batch_gpu(mu_k_base, target_ranks)[0]
    imp_base = feature_importance(genome, D)
    si = MEASURE_NAMES.index("sensitivity")
    log.info("Baseline: r=%.4f, sens_imp=%.3f", r_base, imp_base[si])

    scales = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    results = []

    for scale in scales:
        g = genome.copy()
        g[CANCEL_IDX] *= scale

        mu_knuth = knuth_eval.evaluate_batch(g[None, :])
        r = spearman_batch_gpu(mu_knuth, target_ranks)[0]

        mu_random = random_eval.evaluate_batch(g[None, :])
        bp, bd = check_all_barriers_batch(
            mu_knuth, mu_random, g[None, :], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_violations = sum(1 for v in bd[0].values() if not v["passes"])
        feasible = bool(bp[0])

        median_k = np.median(np.abs(mu_knuth[0]))
        np_density = float(np.mean(np.abs(mu_random[0]) >= median_k))

        margins = {}
        for bname, bdetail in bd[0].items():
            margins[bname] = {
                "passes": bdetail["passes"],
                "density": float(bdetail.get("density", bdetail.get("value", 0))),
            }

        entry = {
            "scale": scale,
            "r": float(r),
            "feasible": feasible,
            "n_violations": n_violations,
            "np_density": np_density,
            "margins": margins,
        }
        results.append(entry)
        log.info("  scale=%.2f | r=%.4f | feas=%s | density=%.4f",
                 scale, r, feasible, np_density)

    out_path = os.path.join(PROJECT_DIR, "data", f"truncation_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved truncation results to %s", out_path)

    _plot_truncation_seed(results, seed_id)
    return results


def _plot_truncation_seed(results, seed_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    scales = [r["scale"] for r in results]
    rs = [r["r"] for r in results]
    densities = [r.get("np_density", 0) for r in results]

    # A: r vs scale
    ax = axes[0]
    ax.plot(scales, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=6)
    ax.axhline(results[-1]["r"], color='gray', linestyle='--', alpha=0.3)
    ax.axhline(results[0]["r"], color='#d32f2f', linestyle='--', alpha=0.3)
    ax.annotate(f'full: r={results[-1]["r"]:.4f}', xy=(0.6, results[-1]["r"]),
                fontsize=8, color='gray')
    ax.annotate(f'no sens: r={results[0]["r"]:.4f}', xy=(0.02, results[0]["r"]+0.0002),
                fontsize=8, color='#d32f2f')
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Spearman r")
    ax.set_title(f"A. Prediction impact (seed {seed_id})")

    # B: NP density vs scale
    ax = axes[1]
    ax.plot(scales, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=6)
    np_thresh = 0.04
    ax.axhline(np_thresh, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.annotate('threshold (1/n²)', xy=(0.5, np_thresh + 0.02), fontsize=8, color='green')
    ax.fill_between(scales, 0, np_thresh, alpha=0.08, color='green')
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Natural-proof density")
    ax.set_title(f"B. Barrier compliance (seed {seed_id})")
    ax.set_ylim(-0.02, 1.05)

    # C: dual-axis overlay
    ax = axes[2]
    ax2 = ax.twinx()
    l1 = ax.plot(scales, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=5, label='Spearman r')
    l2 = ax2.plot(scales, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=5, label='NP density')
    ax2.axhline(np_thresh, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel("Sensitivity parameter scale")
    ax.set_ylabel("Spearman r", color='#1976d2')
    ax2.set_ylabel("NP density", color='#d32f2f')
    ax.set_title(f"C. Tradeoff (seed {seed_id})")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='center left')

    for a in [axes[0], axes[1]]:
        a.grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Progressive truncation: seed {seed_id}", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", f"bics_truncation_seed_{seed_id}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed jump-time analysis — state-controlled vs generation-controlled
# ═══════════════════════════════════════════════════════════════════════════

def multiseed_dynamics(setup_data, max_gen=8000, seeds=None):
    """Run dynamics with multiple seeds, track when sensitivity crosses thresholds.

    Tests whether jumps are triggered by reaching the signal ceiling (state-controlled)
    or occur at fixed generation numbers (generation-controlled).
    """
    if seeds is None:
        seeds = [42, 137, 256, 512, 1024]

    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100
    thresholds = [0.10, 0.30, 0.50]  # sensitivity importance thresholds

    all_results = {}

    for seed in seeds:
        log.info("=" * 60)
        log.info("SEED %d — running %d gens", seed, max_gen)
        log.info("=" * 60)

        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=seed)
        best_genome = None
        best_r = 0.0
        crossing_gens = {t: None for t in thresholds}
        trajectory = []

        genome_path = os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed}.npy")

        for gen in range(max_gen):
            try:
                candidates = es.ask()
                mu_knuth = knuth_eval.evaluate_batch(candidates)
                corr = spearman_batch_gpu(mu_knuth, target_ranks)

                barrier_pass, barrier_details = check_all_barriers_batch(
                    mu_knuth, random_eval.evaluate_batch(candidates),
                    candidates, rel_checker,
                    cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                )
            except (RuntimeError, Exception) as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    log.warning("  seed %d gen %d: CUDA error: %s", seed, gen + 1, e)
                    log.warning("  Saving genome checkpoint and aborting seed %d", seed)
                    if best_genome is not None:
                        np.save(genome_path, best_genome)
                        log.info("  Saved genome at gen %d (r=%.4f)", gen + 1, best_r)
                    # break out of gen loop — save what we have
                    break
                else:
                    raise

            n_violations = np.zeros(pop_size, dtype=np.int32)
            for i in range(pop_size):
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

            gen_num = gen + 1
            if gen_num % 100 == 0 and best_genome is not None:
                imp = feature_importance(best_genome, D)
                si = MEASURE_NAMES.index("sensitivity")
                sens_imp = float(imp[si])

                trajectory.append({
                    "gen": gen_num,
                    "best_r": best_r,
                    "sensitivity_importance": sens_imp,
                    "sigma": float(es.sigma),
                })

                # check threshold crossings
                for t in thresholds:
                    if crossing_gens[t] is None and sens_imp >= t:
                        crossing_gens[t] = gen_num
                        log.info("  seed %d: sens >= %.0f%% at gen %d (r=%.4f)",
                                 seed, t*100, gen_num, best_r)

                if gen_num % 500 == 0:
                    log.info("  seed %d gen %5d | r=%.4f | sens=%.3f | sigma=%.2e",
                             seed, gen_num, best_r, sens_imp, es.sigma)

                # periodic genome save every 1000 gens
                if gen_num % 1000 == 0 and best_genome is not None:
                    np.save(genome_path, best_genome)
                    log.info("  seed %d: checkpoint genome at gen %d (r=%.4f)",
                             seed, gen_num, best_r)

                # periodic GPU cleanup every 500 gens
                if gen_num % 500 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # save genome for later truncation analysis
        if best_genome is not None:
            genome_path = os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed}.npy")
            np.save(genome_path, best_genome)
            log.info("  seed %d: saved genome to %s", seed, genome_path)

        all_results[seed] = {
            "trajectory": trajectory,
            "crossing_gens": crossing_gens,
            "final_r": best_r,
            "final_sens": trajectory[-1]["sensitivity_importance"] if trajectory else 0,
        }
        log.info("  seed %d DONE: r=%.4f, sens=%.3f, crossings=%s",
                 seed, best_r,
                 trajectory[-1]["sensitivity_importance"] if trajectory else 0,
                 crossing_gens)

    # save
    out_path = os.path.join(PROJECT_DIR, "data", "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    log.info("Saved multiseed results to %s", out_path)

    # summary table
    log.info("\n=== JUMP-TIME ALIGNMENT ===")
    log.info("seed  | final_r | final_sens | cross_10%% | cross_30%% | cross_50%%")
    for seed, res in all_results.items():
        cg = res["crossing_gens"]
        log.info("%5d | %.4f  | %.3f      | %s | %s | %s",
                 seed, res["final_r"], res["final_sens"],
                 str(cg.get(0.10, "—")).rjust(5),
                 str(cg.get(0.30, "—")).rjust(5),
                 str(cg.get(0.50, "—")).rjust(5))

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint diagnostics — full basin characterization for a saved genome
# ═══════════════════════════════════════════════════════════════════════════

def endpoint_diagnostics(setup_data, seed_id):
    """Compute full basin-characterization metrics for a saved genome."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    genome_path = os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy")
    if not os.path.exists(genome_path):
        log.error("Genome not found: %s", genome_path)
        sys.exit(1)

    genome = np.load(genome_path)
    log.info("Loaded seed %d genome (%d params)", seed_id, len(genome))

    # basic evaluation
    mu_knuth = knuth_eval.evaluate_batch(genome[None, :])
    r = float(spearman_batch_gpu(mu_knuth, target_ranks)[0])

    mu_random = random_eval.evaluate_batch(genome[None, :])
    bp, bd = check_all_barriers_batch(
        mu_knuth, mu_random, genome[None, :], rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )
    feasible = bool(bp[0])
    n_violations = sum(1 for v in bd[0].values() if not v["passes"])

    # NP density
    median_k = np.median(np.abs(mu_knuth[0]))
    np_density = float(np.mean(np.abs(mu_random[0]) >= median_k))

    # per-barrier margins
    barrier_status = {}
    for bname, bdetail in bd[0].items():
        barrier_status[bname] = {
            "passes": bdetail["passes"],
            "density": float(bdetail.get("density", bdetail.get("value", 0))),
        }

    # sensitivity importance
    imp = feature_importance(genome, D)
    si = MEASURE_NAMES.index("sensitivity")
    sens_imp = float(imp[si])

    # weight subspace norms
    cancel_l1 = float(np.abs(genome[CANCEL_IDX]).sum())
    signal_l1 = float(np.abs(genome[SIGNAL_IDX]).sum())
    total_l1 = cancel_l1 + signal_l1
    cancel_frac = cancel_l1 / (total_l1 + 1e-12)

    # cross-term mass: quadratic terms involving sensitivity
    q_start = D
    cross_term_mass = 0.0
    total_quad_mass = 0.0
    k = 0
    for i in range(D):
        for j in range(i, D):
            w = abs(genome[q_start + k])
            total_quad_mass += w
            if i == si or j == si:
                cross_term_mass += w
            k += 1
    cross_term_frac = cross_term_mass / (total_quad_mass + 1e-12)

    result = {
        "seed": seed_id,
        "r": r,
        "feasible": feasible,
        "n_violations": n_violations,
        "np_density": np_density,
        "sensitivity_importance": sens_imp,
        "cancel_weight_frac": cancel_frac,
        "cancel_l1": cancel_l1,
        "signal_l1": signal_l1,
        "cross_term_mass": cross_term_mass,
        "cross_term_frac": cross_term_frac,
        "total_quad_mass": total_quad_mass,
        "barrier_status": barrier_status,
    }

    log.info("=== ENDPOINT DIAGNOSTICS: seed %d ===", seed_id)
    log.info("  r = %.4f", r)
    log.info("  feasible = %s (violations: %d)", feasible, n_violations)
    log.info("  NP density = %.4f", np_density)
    log.info("  sens_importance = %.3f", sens_imp)
    log.info("  cancel_weight_frac = %.3f", cancel_frac)
    log.info("  cross_term_frac = %.3f", cross_term_frac)
    for bname, bs in barrier_status.items():
        log.info("  barrier %s: passes=%s, density=%.4f", bname, bs["passes"], bs["density"])

    out_path = os.path.join(PROJECT_DIR, "data", f"endpoint_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved to %s", out_path)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Genome interpolation — basin connectivity test
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_genomes(setup_data, seed_a, seed_b, n_steps=21):
    """Linear interpolation between two seed genomes to test basin connectivity."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    g_a = np.load(os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_a}.npy"))
    g_b = np.load(os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_b}.npy"))
    log.info("Interpolating seed %d <-> seed %d (%d steps)", seed_a, seed_b, n_steps)

    alphas = np.linspace(0.0, 1.0, n_steps)
    results = []

    for alpha in alphas:
        g = (1 - alpha) * g_a + alpha * g_b

        mu_knuth = knuth_eval.evaluate_batch(g[None, :])
        r = float(spearman_batch_gpu(mu_knuth, target_ranks)[0])

        mu_random = random_eval.evaluate_batch(g[None, :])
        bp, bd = check_all_barriers_batch(
            mu_knuth, mu_random, g[None, :], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        feasible = bool(bp[0])
        n_violations = sum(1 for v in bd[0].values() if not v["passes"])

        median_k = np.median(np.abs(mu_knuth[0]))
        np_density = float(np.mean(np.abs(mu_random[0]) >= median_k))

        barrier_status = {}
        for bname, bdetail in bd[0].items():
            barrier_status[bname] = {
                "passes": bdetail["passes"],
                "density": float(bdetail.get("density", bdetail.get("value", 0))),
            }

        entry = {
            "alpha": float(alpha),
            "r": r,
            "feasible": feasible,
            "n_violations": n_violations,
            "np_density": np_density,
            "barrier_status": barrier_status,
        }
        results.append(entry)
        log.info("  alpha=%.2f | r=%.4f | feas=%s | np_density=%.4f | viol=%d",
                 alpha, r, feasible, np_density, n_violations)

    out_path = os.path.join(PROJECT_DIR, "data",
                            f"interpolation_{seed_a}_{seed_b}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_interpolation(results, seed_a, seed_b)
    return results


def _plot_interpolation(results, seed_a, seed_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [e["alpha"] for e in results]
    rs = [e["r"] for e in results]
    densities = [e["np_density"] for e in results]
    feasible = [e["feasible"] for e in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # A: r along interpolation path
    ax = axes[0]
    ax.plot(alphas, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=5)
    ax.axhline(rs[0], color='#1976d2', linestyle='--', alpha=0.3)
    ax.axhline(rs[-1], color='#d32f2f', linestyle='--', alpha=0.3)
    ax.annotate(f'seed {seed_a}: r={rs[0]:.4f}', xy=(0.02, rs[0]), fontsize=7, va='bottom')
    ax.annotate(f'seed {seed_b}: r={rs[-1]:.4f}', xy=(0.7, rs[-1]), fontsize=7, va='bottom')
    ax.set_xlabel(f"α (0=seed {seed_a}, 1=seed {seed_b})")
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Prediction along path")
    ax.grid(True, alpha=0.3)

    # B: NP density along path
    ax = axes[1]
    ax.plot(alphas, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=5)
    ax.axhline(0.04, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.fill_between(alphas, 0, 0.04, alpha=0.06, color='green')
    ax.set_xlabel(f"α (0=seed {seed_a}, 1=seed {seed_b})")
    ax.set_ylabel("NP density")
    ax.set_title("B. NP density along path")
    ax.grid(True, alpha=0.3)

    # C: feasibility + barrier status
    ax = axes[2]
    colors = []
    for e in results:
        if e["feasible"]:
            colors.append('#4caf50')
        else:
            np_fail = not e["barrier_status"].get("natural_proof", {}).get("passes", True)
            alg_fail = not e["barrier_status"].get("algebrization", {}).get("passes", True)
            if np_fail and alg_fail:
                colors.append('#9c27b0')  # both fail
            elif np_fail:
                colors.append('#d32f2f')  # NP fail
            elif alg_fail:
                colors.append('#ff9800')  # alg fail
            else:
                colors.append('#607d8b')  # other
    ax.bar(alphas, [1]*len(alphas), width=0.04, color=colors, alpha=0.8)
    ax.set_yticks([])
    ax.set_xlabel(f"α (0=seed {seed_a}, 1=seed {seed_b})")
    ax.set_title("C. Barrier status")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4caf50', alpha=0.8, label='All pass'),
        Patch(facecolor='#d32f2f', alpha=0.8, label='NP fails'),
        Patch(facecolor='#ff9800', alpha=0.8, label='Alg fails'),
        Patch(facecolor='#9c27b0', alpha=0.8, label='Both fail'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    fig.suptitle(f"Basin connectivity: seed {seed_a} ↔ seed {seed_b}",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures",
                            f"interpolation_{seed_a}_{seed_b}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Basin volume probe — random perturbation around a genome
# ═══════════════════════════════════════════════════════════════════════════

def basin_volume(setup_data, seed_id, n_samples=50):
    """Probe basin volume by adding random perturbations at increasing scales."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    genome = np.load(os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy"))
    log.info("Basin volume probe: seed %d (%d params)", seed_id, len(genome))

    genome_norm = float(np.linalg.norm(genome))
    log.info("  genome L2 norm = %.4f", genome_norm)

    sigmas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    rng = np.random.RandomState(42)
    results = []

    for sigma in sigmas:
        rs, densities, feas_count = [], [], 0
        for _ in range(n_samples):
            g = genome + rng.randn(len(genome)) * sigma
            mu_k = knuth_eval.evaluate_batch(g[None, :])
            r = float(spearman_batch_gpu(mu_k, target_ranks)[0])
            mu_r = random_eval.evaluate_batch(g[None, :])
            bp, bd = check_all_barriers_batch(
                mu_k, mu_r, g[None, :], rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )
            median_k = np.median(np.abs(mu_k[0]))
            np_d = float(np.mean(np.abs(mu_r[0]) >= median_k))

            rs.append(r)
            densities.append(np_d)
            if bp[0]:
                feas_count += 1

        entry = {
            "sigma": sigma,
            "r_mean": float(np.mean(rs)),
            "r_std": float(np.std(rs)),
            "r_min": float(np.min(rs)),
            "density_mean": float(np.mean(densities)),
            "density_std": float(np.std(densities)),
            "feas_frac": feas_count / n_samples,
        }
        results.append(entry)
        log.info("  sigma=%.3f | r=%.4f+/-%.4f | density=%.4f | feas=%.0f%%",
                 sigma, entry["r_mean"], entry["r_std"],
                 entry["density_mean"], entry["feas_frac"]*100)

    out_path = os.path.join(PROJECT_DIR, "data", f"basin_volume_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_basin_volume(results, seed_id)
    return results


def _plot_basin_volume(results, seed_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = [e["sigma"] for e in results]
    r_means = [e["r_mean"] for e in results]
    r_stds = [e["r_std"] for e in results]
    d_means = [e["density_mean"] for e in results]
    feas = [e["feas_frac"] for e in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.errorbar(sigmas, r_means, yerr=r_stds, fmt='o-', color='#1976d2',
                linewidth=1.5, markersize=5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel("Perturbation sigma")
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Prediction stability")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sigmas, d_means, 's-', color='#d32f2f', linewidth=1.5, markersize=5)
    ax.axhline(0.04, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.fill_between(sigmas, 0, 0.04, alpha=0.06, color='green')
    ax.set_xscale('log')
    ax.set_xlabel("Perturbation sigma")
    ax.set_ylabel("NP density")
    ax.set_title("B. NP density under perturbation")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(sigmas, [f * 100 for f in feas], 'D-', color='#4caf50',
            linewidth=1.5, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel("Perturbation sigma")
    ax.set_ylabel("Feasible fraction (%)")
    ax.set_title("C. Basin feasibility radius")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Basin volume probe: seed {seed_id}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", f"basin_volume_seed_{seed_id}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity injection — progressively add sensitivity to a pure-signal genome
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_injection(setup_data, seed_id):
    """Take a genome and progressively inject sensitivity cross-term weight.

    Unlike truncation (which scales existing sensitivity), this ADDS new
    sensitivity weight on top of the existing genome to test when barriers break.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    genome = np.load(os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy"))
    log.info("Sensitivity injection: seed %d", seed_id)

    # reference: seed 42's sensitivity params as injection template
    g42 = np.load(os.path.join(PROJECT_DIR, "data", "genome_seed_42.npy"))
    sens_template = np.zeros_like(g42)
    sens_template[CANCEL_IDX] = g42[CANCEL_IDX]
    template_norm = float(np.linalg.norm(sens_template))
    log.info("  injection template from seed 42, L2=%.4f", template_norm)

    scales = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]
    results = []

    for scale in scales:
        g = genome.copy()
        g[CANCEL_IDX] += scale * g42[CANCEL_IDX]

        mu_k = knuth_eval.evaluate_batch(g[None, :])
        r = float(spearman_batch_gpu(mu_k, target_ranks)[0])
        mu_r = random_eval.evaluate_batch(g[None, :])
        bp, bd = check_all_barriers_batch(
            mu_k, mu_r, g[None, :], rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        feasible = bool(bp[0])
        median_k = np.median(np.abs(mu_k[0]))
        np_density = float(np.mean(np.abs(mu_r[0]) >= median_k))

        barrier_status = {}
        for bname, bdetail in bd[0].items():
            barrier_status[bname] = {
                "passes": bdetail["passes"],
                "density": float(bdetail.get("density", bdetail.get("value", 0))),
            }

        imp = feature_importance(g, D)
        si = MEASURE_NAMES.index("sensitivity")
        sens_imp = float(imp[si])

        entry = {
            "scale": scale,
            "r": r,
            "feasible": feasible,
            "np_density": np_density,
            "sensitivity_importance": sens_imp,
            "barrier_status": barrier_status,
        }
        results.append(entry)
        log.info("  inject=%.2f | r=%.4f | feas=%s | density=%.4f | sens=%.3f",
                 scale, r, feasible, np_density, sens_imp)

    out_path = os.path.join(PROJECT_DIR, "data", f"injection_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_injection(results, seed_id)
    return results


def _plot_injection(results, seed_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = [e["scale"] for e in results]
    rs = [e["r"] for e in results]
    densities = [e["np_density"] for e in results]
    sens_imps = [e["sensitivity_importance"] for e in results]
    feasible = [e["feasible"] for e in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(scales, rs, 'o-', color='#1976d2', linewidth=1.5, markersize=5)
    for i, f in enumerate(feasible):
        if not f:
            ax.plot(scales[i], rs[i], 'x', color='#d32f2f', markersize=10, zorder=10)
    ax.set_xlabel("Injection scale (fraction of seed 42's sensitivity)")
    ax.set_ylabel("Spearman r")
    ax.set_title("A. Prediction under injection")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(scales, densities, 's-', color='#d32f2f', linewidth=1.5, markersize=5)
    ax.axhline(0.04, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.fill_between(scales, 0, 0.04, alpha=0.06, color='green')
    ax.set_xlabel("Injection scale")
    ax.set_ylabel("NP density")
    ax.set_title("B. NP compliance under injection")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(scales, sens_imps, '^-', color='#ff9800', linewidth=1.5, markersize=5)
    ax.set_xlabel("Injection scale")
    ax.set_ylabel("Sensitivity importance")
    ax.set_title("C. Sensitivity importance growth")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity injection into seed {seed_id} (template: seed 42)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", f"injection_seed_{seed_id}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Genome classification helper
# ═══════════════════════════════════════════════════════════════════════════

def classify_genome(genome, knuth_eval, random_eval, target_ranks, rel_checker, cfg):
    """Classify a genome into Type A/B/C and return metrics."""
    mu_k = knuth_eval.evaluate_batch(genome[None, :])
    r = float(spearman_batch_gpu(mu_k, target_ranks)[0])

    mu_r = random_eval.evaluate_batch(genome[None, :])
    bp, bd = check_all_barriers_batch(
        mu_k, mu_r, genome[None, :], rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )
    feasible = bool(bp[0])

    median_k = np.median(np.abs(mu_k[0]))
    np_density = float(np.mean(np.abs(mu_r[0]) >= median_k))

    imp = feature_importance(genome, D)
    si = MEASURE_NAMES.index("sensitivity")
    sens_imp = float(imp[si])

    # cross-term fraction
    q_start = D
    cross_mass = 0.0
    total_q = 0.0
    k = 0
    for i in range(D):
        for j in range(i, D):
            w = abs(genome[q_start + k])
            total_q += w
            if i == si or j == si:
                cross_mass += w
            k += 1
    cross_frac = cross_mass / (total_q + 1e-12)

    # classify: thresholds from 5-seed empirical data
    if sens_imp > 0.35 and cross_frac > 0.40:
        regime = "A"
    elif sens_imp < 0.05 and cross_frac < 0.05:
        regime = "C"
    else:
        regime = "B"

    return {
        "r": r, "feasible": feasible, "np_density": np_density,
        "sensitivity_importance": sens_imp, "cross_term_frac": cross_frac,
        "regime": regime,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Local continuation — is Type C a basin or a spike?
# ═══════════════════════════════════════════════════════════════════════════

def local_continuation(setup_data, seed_id=512, n_per_sigma=10, refine_gens=100):
    """Perturb a genome and do short CMA-ES refinement to test basin recovery.

    If perturbed points recover to high-r feasible solutions with low overhead,
    Type C is a real basin. If not, it is just a narrow spike.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 50

    genome = np.load(os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy"))
    log.info("Local continuation: seed %d, n_per_sigma=%d, refine_gens=%d",
             seed_id, n_per_sigma, refine_gens)

    baseline = classify_genome(genome, knuth_eval, random_eval,
                               target_ranks, rel_checker, cfg)
    log.info("  baseline: r=%.4f regime=%s sens=%.3f cross=%.3f",
             baseline["r"], baseline["regime"],
             baseline["sensitivity_importance"], baseline["cross_term_frac"])

    sigmas = [0.005, 0.01, 0.02, 0.05]
    rng = np.random.RandomState(42)
    results = {"baseline": baseline, "sigmas": []}

    for sigma in sigmas:
        log.info("  === sigma=%.3f ===", sigma)
        sigma_results = {"sigma": sigma, "points": []}

        for idx in range(n_per_sigma):
            perturbed = genome + rng.randn(len(genome)) * sigma

            pre = classify_genome(perturbed, knuth_eval, random_eval,
                                  target_ranks, rel_checker, cfg)

            # short CMA-ES refinement from perturbed point
            es = CMAES(GENOME_DIM, sigma0=sigma * 0.5, sigma_max=100.0,
                       sigma_min=1e-8, pop_size=pop_size, seed=1000 + idx)
            es.mean = perturbed.copy()

            best_genome = perturbed.copy()
            best_r = pre["r"]

            for gen in range(refine_gens):
                try:
                    candidates = es.ask()
                    mu_k = knuth_eval.evaluate_batch(candidates)
                    corr = spearman_batch_gpu(mu_k, target_ranks)
                    bp, bd = check_all_barriers_batch(
                        mu_k, random_eval.evaluate_batch(candidates),
                        candidates, rel_checker,
                        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                    )
                except (RuntimeError, Exception) as e:
                    if "cuda" in str(e).lower():
                        log.warning("    CUDA error: %s", e)
                        break
                    raise

                nv = np.zeros(pop_size, dtype=np.int32)
                for j in range(pop_size):
                    if not bp[j]:
                        nv[j] = sum(1 for v in bd[j].values() if not v["passes"])

                l2 = np.sqrt((candidates ** 2).sum(axis=1))
                fit = -corr + penalty * nv + l2_lambda * l2
                es.tell(candidates, fit)
                if es.should_restart():
                    es.restart()

                ib = corr.argmax()
                if corr[ib] > best_r:
                    best_r = float(corr[ib])
                    best_genome = candidates[ib].copy()

            post = classify_genome(best_genome, knuth_eval, random_eval,
                                   target_ranks, rel_checker, cfg)

            point = {
                "idx": idx,
                "pre_r": pre["r"],
                "pre_feasible": pre["feasible"],
                "pre_regime": pre["regime"],
                "pre_sens": pre["sensitivity_importance"],
                "post_r": post["r"],
                "post_feasible": post["feasible"],
                "post_regime": post["regime"],
                "post_sens": post["sensitivity_importance"],
                "post_cross": post["cross_term_frac"],
                "recovered": post["r"] > 0.55 and post["feasible"],
            }
            sigma_results["points"].append(point)

            log.info("    [%d/%d] pre: r=%.4f %s -> post: r=%.4f %s regime=%s %s",
                     idx + 1, n_per_sigma,
                     pre["r"], "F" if pre["feasible"] else "X",
                     post["r"], "F" if post["feasible"] else "X",
                     post["regime"],
                     "RECOVERED" if point["recovered"] else "")

        n_recovered = sum(1 for p in sigma_results["points"] if p["recovered"])
        regimes = {"A": 0, "B": 0, "C": 0}
        for p in sigma_results["points"]:
            if p["recovered"]:
                regimes[p["post_regime"]] += 1
        sigma_results["recovery_rate"] = n_recovered / n_per_sigma
        sigma_results["recovered_regimes"] = regimes
        results["sigmas"].append(sigma_results)

        log.info("  sigma=%.3f: recovered %d/%d (%.0f%%) -- A=%d B=%d C=%d",
                 sigma, n_recovered, n_per_sigma, 100 * n_recovered / n_per_sigma,
                 regimes["A"], regimes["B"], regimes["C"])

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    out_path = os.path.join(PROJECT_DIR, "data",
                            f"local_continuation_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_local_continuation(results, seed_id)
    return results


def _plot_local_continuation(results, seed_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sigma_data = results["sigmas"]
    sigmas = [s["sigma"] for s in sigma_data]

    # A: Recovery rate vs sigma
    ax = axes[0, 0]
    rates = [s["recovery_rate"] * 100 for s in sigma_data]
    ax.bar(range(len(sigmas)), rates, color='#4caf50', alpha=0.8,
           tick_label=[f'{s:.3f}' for s in sigmas])
    ax.set_xlabel("Perturbation sigma")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_title("A. Recovery rate (r>0.55 + feasible)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # B: Pre vs Post r scatter
    ax = axes[0, 1]
    for sd in sigma_data:
        for p in sd["points"]:
            color = '#4caf50' if p["recovered"] else '#d32f2f'
            ax.scatter(p["pre_r"], p["post_r"], c=color, s=40, alpha=0.6,
                       edgecolors='black', linewidth=0.3)
    ax.plot([0, 0.7], [0, 0.7], '--', color='gray', alpha=0.5)
    ax.axhline(0.55, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel("Pre-refinement r")
    ax.set_ylabel("Post-refinement r")
    ax.set_title("B. Refinement recovery")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#4caf50', label='Recovered'),
        Patch(color='#d32f2f', label='Not recovered'),
    ], fontsize=8)

    # C: Regime distribution of recovered points
    ax = axes[1, 0]
    regime_colors = {"A": "#1976d2", "B": "#ff9800", "C": "#4caf50"}
    x = np.arange(len(sigmas))
    width = 0.25
    for ri, regime in enumerate(["A", "B", "C"]):
        counts = [sd["recovered_regimes"].get(regime, 0) for sd in sigma_data]
        ax.bar(x + ri * width, counts, width, label=f"Type {regime}",
               color=regime_colors[regime], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{s:.3f}' for s in sigmas], fontsize=9)
    ax.set_xlabel("Perturbation sigma")
    ax.set_ylabel("Count (recovered only)")
    ax.set_title("C. Regime of recovered points")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # D: Post-refinement fingerprint of recovered points
    ax = axes[1, 1]
    for sd in sigma_data:
        for p in sd["points"]:
            if p["recovered"]:
                regime = p["post_regime"]
                color = regime_colors.get(regime, '#607d8b')
                ax.scatter(p["post_sens"], p["post_cross"], c=color,
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    bl = results["baseline"]
    ax.scatter(bl["sensitivity_importance"], bl["cross_term_frac"],
               c='red', marker='*', s=200, zorder=10,
               label=f'seed {seed_id} baseline')
    ax.set_xlabel("Sensitivity importance")
    ax.set_ylabel("Cross-term fraction")
    ax.set_title("D. Recovered endpoints fingerprint")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Local continuation: seed {seed_id} basin structure",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures",
                            f"local_continuation_seed_{seed_id}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Targeted recovery — overhead-penalized search for Type C
# ═══════════════════════════════════════════════════════════════════════════

def targeted_recovery(setup_data, n_seeds=10, max_gen=4000, lambda_oh=0.05):
    """Test whether overhead penalty can increase Type C discovery rate.

    Three conditions:
      standard:   fitness = -corr + barrier_penalty * violations + l2 * norm
      penalized:  + lambda_oh * cancel_l1 (penalize sensitivity mass)
      warmstart:  penalized + init from Type B genome (seed 137)
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100

    ws_path = os.path.join(PROJECT_DIR, "data", "genome_seed_137.npy")
    warm_genome = np.load(ws_path) if os.path.exists(ws_path) else None

    conditions = ["standard", "penalized"]
    if warm_genome is not None:
        conditions.append("warmstart")

    all_results = {}
    seed_base = 2000

    for cond in conditions:
        log.info("=" * 60)
        log.info("TARGETED RECOVERY -- %s (n=%d, max_gen=%d, lambda_oh=%.3f)",
                 cond, n_seeds, max_gen, lambda_oh)
        log.info("=" * 60)

        cond_results = []

        for i in range(n_seeds):
            seed = seed_base + i
            log.info("  --- %s seed %d (%d/%d) ---", cond, seed, i + 1, n_seeds)

            es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                       pop_size=pop_size, seed=seed)

            if cond == "warmstart" and warm_genome is not None:
                es.mean = warm_genome.copy()
                es.sigma = 0.5

            best_genome = None
            best_r = 0.0

            for gen in range(max_gen):
                try:
                    candidates = es.ask()
                    mu_knuth = knuth_eval.evaluate_batch(candidates)
                    corr = spearman_batch_gpu(mu_knuth, target_ranks)

                    bp, bd = check_all_barriers_batch(
                        mu_knuth, random_eval.evaluate_batch(candidates),
                        candidates, rel_checker,
                        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                    )
                except (RuntimeError, Exception) as e:
                    if "cuda" in str(e).lower():
                        log.warning("  CUDA error at gen %d: %s", gen + 1, e)
                        break
                    raise

                n_violations = np.zeros(pop_size, dtype=np.int32)
                for j in range(pop_size):
                    if not bp[j]:
                        n_violations[j] = sum(
                            1 for v in bd[j].values() if not v["passes"])

                l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
                fitness = -corr + penalty * n_violations + l2_lambda * l2_norm

                if cond in ("penalized", "warmstart"):
                    cancel_l1 = np.abs(candidates[:, CANCEL_IDX]).sum(axis=1)
                    fitness += lambda_oh * cancel_l1

                es.tell(candidates, fitness)
                if es.should_restart():
                    es.restart()

                idx_best = corr.argmax()
                if corr[idx_best] > best_r:
                    best_r = float(corr[idx_best])
                    best_genome = candidates[idx_best].copy()

                if (gen + 1) % 1000 == 0:
                    imp = feature_importance(best_genome, D)
                    si = MEASURE_NAMES.index("sensitivity")
                    log.info("    gen %4d | r=%.4f | sens=%.3f | sigma=%.2e",
                             gen + 1, best_r, float(imp[si]), es.sigma)

                if (gen + 1) % 500 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            if best_genome is not None:
                cls = classify_genome(best_genome, knuth_eval, random_eval,
                                      target_ranks, rel_checker, cfg)
                cls["seed"] = seed
                cond_results.append(cls)
                log.info("  seed %d: r=%.4f feas=%s regime=%s sens=%.3f cross=%.3f",
                         seed, cls["r"], cls["feasible"], cls["regime"],
                         cls["sensitivity_importance"], cls["cross_term_frac"])

        all_results[cond] = cond_results

        counts = {"A": 0, "B": 0, "C": 0}
        for c in cond_results:
            counts[c["regime"]] += 1
        log.info("  %s regime distribution: A=%d B=%d C=%d",
                 cond, counts["A"], counts["B"], counts["C"])

    out_path = os.path.join(PROJECT_DIR, "data", "targeted_recovery.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_targeted_recovery(all_results)
    return all_results


def _plot_targeted_recovery(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cond_colors = {"standard": "#607d8b", "penalized": "#1976d2", "warmstart": "#4caf50"}
    regime_colors = {"A": "#1976d2", "B": "#ff9800", "C": "#4caf50"}

    # A: regime distribution per condition
    ax = axes[0]
    conds = list(all_results.keys())
    x = np.arange(len(conds))
    width = 0.25
    for ri, regime in enumerate(["A", "B", "C"]):
        counts = [sum(1 for c in all_results[cond] if c["regime"] == regime)
                  for cond in conds]
        ax.bar(x + ri * width, counts, width, label=f"Type {regime}",
               color=regime_colors[regime], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("A. Regime distribution by condition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # B: endpoint scatter
    ax = axes[1]
    from matplotlib.patches import Patch
    for cond in conds:
        for c in all_results[cond]:
            marker = 'D' if c["regime"] == "A" else 's' if c["regime"] == "B" else '*'
            size = 100 if c["regime"] in ("A", "C") else 60
            ax.scatter(c["sensitivity_importance"], c["cross_term_frac"],
                       c=cond_colors[cond], marker=marker, s=size, alpha=0.7,
                       edgecolors='black', linewidth=0.3)
    ax.set_xlabel("Sensitivity importance")
    ax.set_ylabel("Cross-term fraction")
    ax.set_title("B. Endpoint fingerprints")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[Patch(color=cond_colors[c], label=c) for c in conds],
              fontsize=8, loc='upper left')

    # C: final r box plot
    ax = axes[2]
    bp_data = [[c["r"] for c in all_results[cond]] for cond in conds]
    bp = ax.boxplot(bp_data, labels=conds, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cond_colors[conds[i]])
        patch.set_alpha(0.6)
    ax.set_ylabel("Final Spearman r")
    ax.set_title("C. Prediction quality")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Targeted recovery: overhead penalty effect on basin selection",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", "targeted_recovery.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Irreversibility test — can Type B return to Type C?
# ═══════════════════════════════════════════════════════════════════════════

def irreversibility_test(setup_data, seed_ids=None, refine_gens=300, n_trials=5):
    """From Type B genomes, run short standard CMA-ES (no overhead penalty).

    If Type B can return to Type C → "default path problem" (moderate claim).
    If Type B cannot return    → "one-way degradation" (strong claim).
    """
    if seed_ids is None:
        seed_ids = [137, 256, 1024]

    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 50

    all_results = {}

    for sid in seed_ids:
        gpath = os.path.join(PROJECT_DIR, "data", f"genome_seed_{sid}.npy")
        if not os.path.exists(gpath):
            log.warning("  genome for seed %d not found, skipping", sid)
            continue

        genome = np.load(gpath)
        baseline = classify_genome(genome, knuth_eval, random_eval,
                                   target_ranks, rel_checker, cfg)
        log.info("=" * 60)
        log.info("IRREVERSIBILITY TEST — seed %d (baseline: regime=%s r=%.4f sens=%.3f cross=%.3f feas=%s)",
                 sid, baseline["regime"], baseline["r"],
                 baseline["sensitivity_importance"], baseline["cross_term_frac"],
                 baseline["feasible"])
        log.info("=" * 60)

        trials = []
        for trial in range(n_trials):
            log.info("  --- trial %d/%d (sigma0=0.01, %d gens, NO penalty) ---",
                     trial + 1, n_trials, refine_gens)

            es = CMAES(GENOME_DIM, sigma0=0.01, sigma_max=100.0,
                       sigma_min=1e-8, pop_size=pop_size, seed=5000 + sid * 100 + trial)
            es.mean = genome.copy()

            best_genome = genome.copy()
            best_r = baseline["r"]

            for gen in range(refine_gens):
                try:
                    candidates = es.ask()
                    mu_k = knuth_eval.evaluate_batch(candidates)
                    corr = spearman_batch_gpu(mu_k, target_ranks)
                    bp, bd = check_all_barriers_batch(
                        mu_k, random_eval.evaluate_batch(candidates),
                        candidates, rel_checker,
                        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                    )
                except (RuntimeError, Exception) as e:
                    if "cuda" in str(e).lower():
                        log.warning("    CUDA error: %s", e)
                        break
                    raise

                nv = np.zeros(pop_size, dtype=np.int32)
                for j in range(pop_size):
                    if not bp[j]:
                        nv[j] = sum(1 for v in bd[j].values() if not v["passes"])

                l2 = np.sqrt((candidates ** 2).sum(axis=1))
                fit = -corr + penalty * nv + l2_lambda * l2
                es.tell(candidates, fit)
                if es.should_restart():
                    es.restart()

                ib = corr.argmax()
                if corr[ib] > best_r:
                    best_r = float(corr[ib])
                    best_genome = candidates[ib].copy()

                if (gen + 1) % 100 == 0:
                    imp = feature_importance(best_genome, D)
                    si_idx = MEASURE_NAMES.index("sensitivity")
                    log.info("    gen %3d | r=%.4f | sens=%.3f | sigma=%.2e",
                             gen + 1, best_r, float(imp[si_idx]), es.sigma)

            post = classify_genome(best_genome, knuth_eval, random_eval,
                                   target_ranks, rel_checker, cfg)

            returned_to_c = post["regime"] == "C"
            trial_result = {
                "trial": trial,
                "pre_regime": baseline["regime"],
                "pre_r": baseline["r"],
                "pre_sens": baseline["sensitivity_importance"],
                "pre_cross": baseline["cross_term_frac"],
                "pre_feasible": baseline["feasible"],
                "post_regime": post["regime"],
                "post_r": post["r"],
                "post_sens": post["sensitivity_importance"],
                "post_cross": post["cross_term_frac"],
                "post_feasible": post["feasible"],
                "returned_to_c": returned_to_c,
            }
            trials.append(trial_result)
            log.info("  trial %d: %s->%s  r=%.4f->%.4f  sens=%.3f->%.3f  %s",
                     trial + 1, baseline["regime"], post["regime"],
                     baseline["r"], post["r"],
                     baseline["sensitivity_importance"], post["sensitivity_importance"],
                     "RETURNED" if returned_to_c else "STUCK")

        n_returned = sum(1 for t in trials if t["returned_to_c"])
        log.info("  seed %d summary: %d/%d returned to Type C", sid, n_returned, n_trials)
        all_results[str(sid)] = {"baseline": baseline, "trials": trials,
                                 "n_returned": n_returned}

    out_path = os.path.join(PROJECT_DIR, "data", "irreversibility_test.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out_path)

    total_returned = sum(v["n_returned"] for v in all_results.values())
    total_trials = sum(len(v["trials"]) for v in all_results.values())
    log.info("OVERALL: %d/%d returned to Type C", total_returned, total_trials)
    if total_returned == 0:
        log.info("CONCLUSION: irreversible — one-way degradation (strong claim)")
    elif total_returned == total_trials:
        log.info("CONCLUSION: fully reversible — default path problem (moderate claim)")
    else:
        log.info("CONCLUSION: partially reversible — seed-dependent (nuanced claim)")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Sensitivity ablation — which CANCEL_IDX dimensions are critical?
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_ablation(setup_data, seed_id=512):
    """Zero each CANCEL_IDX dimension individually on a Type C genome.

    Identifies the 'critical few' dimensions whose removal breaks feasibility,
    revealing that feasibility depends on structured micro-distribution, not
    total sensitivity mass.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    gpath = os.path.join(PROJECT_DIR, "data", f"genome_seed_{seed_id}.npy")
    genome = np.load(gpath)

    baseline = classify_genome(genome, knuth_eval, random_eval,
                               target_ranks, rel_checker, cfg)
    log.info("=" * 60)
    log.info("SENSITIVITY ABLATION — seed %d", seed_id)
    log.info("  baseline: regime=%s r=%.4f sens=%.3f cross=%.3f feas=%s np_d=%.4f",
             baseline["regime"], baseline["r"],
             baseline["sensitivity_importance"], baseline["cross_term_frac"],
             baseline["feasible"], baseline["np_density"])
    log.info("=" * 60)

    si = MEASURE_NAMES.index("sensitivity")
    dim_labels = _cancel_dim_labels(si)

    results = {"seed": seed_id, "baseline": baseline, "ablations": []}

    log.info("  %-4s %-28s %8s %8s %8s %8s %5s %6s",
             "idx", "dimension", "orig_val", "r", "sens", "cross", "feas", "regime")
    log.info("  " + "-" * 95)

    for ci, cancel_dim in enumerate(CANCEL_IDX):
        ablated = genome.copy()
        orig_val = float(ablated[cancel_dim])
        ablated[cancel_dim] = 0.0

        cls = classify_genome(ablated, knuth_eval, random_eval,
                              target_ranks, rel_checker, cfg)

        label = dim_labels[ci] if ci < len(dim_labels) else f"dim_{cancel_dim}"
        entry = {
            "cancel_idx_pos": ci,
            "genome_dim": cancel_dim,
            "label": label,
            "original_value": orig_val,
            "r": cls["r"],
            "feasible": cls["feasible"],
            "np_density": cls["np_density"],
            "sensitivity_importance": cls["sensitivity_importance"],
            "cross_term_frac": cls["cross_term_frac"],
            "regime": cls["regime"],
            "r_delta": cls["r"] - baseline["r"],
            "feasibility_lost": baseline["feasible"] and not cls["feasible"],
        }
        results["ablations"].append(entry)

        feas_mark = "F" if cls["feasible"] else "X"
        lost_mark = " LOST!" if entry["feasibility_lost"] else ""
        log.info("  [%2d] %-28s %+8.4f %8.4f %8.3f %8.3f %5s %6s%s",
                 ci, label, orig_val, cls["r"],
                 cls["sensitivity_importance"], cls["cross_term_frac"],
                 feas_mark, cls["regime"], lost_mark)

    # zero ALL cancel dims at once
    all_zeroed = genome.copy()
    all_zeroed[CANCEL_IDX] = 0.0
    cls_all = classify_genome(all_zeroed, knuth_eval, random_eval,
                              target_ranks, rel_checker, cfg)
    results["all_zeroed"] = {
        "r": cls_all["r"],
        "feasible": cls_all["feasible"],
        "np_density": cls_all["np_density"],
        "sensitivity_importance": cls_all["sensitivity_importance"],
        "cross_term_frac": cls_all["cross_term_frac"],
        "regime": cls_all["regime"],
    }
    log.info("")
    log.info("  ALL zeroed: r=%.4f sens=%.3f cross=%.3f feas=%s regime=%s",
             cls_all["r"], cls_all["sensitivity_importance"],
             cls_all["cross_term_frac"], cls_all["feasible"], cls_all["regime"])

    critical = [a for a in results["ablations"] if a["feasibility_lost"]]
    results["n_critical"] = len(critical)
    log.info("")
    log.info("  Critical dimensions (feasibility lost): %d / %d", len(critical), len(CANCEL_IDX))
    for c in critical:
        log.info("    [%d] %s (val=%.4f)", c["cancel_idx_pos"], c["label"], c["original_value"])

    out_path = os.path.join(PROJECT_DIR, "data", f"sensitivity_ablation_seed_{seed_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    return results


def _cancel_dim_labels(si):
    """Human-readable labels for each CANCEL_IDX dimension."""
    labels = [f"W[{si}] (linear weight)"]
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                if i == j:
                    labels.append(f"Q[{i},{j}] (sensitivity^2)")
                else:
                    other = j if i == si else i
                    labels.append(f"Q[{i},{j}] ({MEASURE_NAMES[other]}*sens)")
            k += 1
    labels.append(f"A[{si}] (activation)")
    labels.append(f"T[{si}] (threshold)")
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Covariance mechanism test — is covariance the lock?
# ═══════════════════════════════════════════════════════════════════════════

def covariance_mechanism_test(setup_data, seed_ids=None, refine_gens=300, n_trials=3):
    """Diagnose whether covariance adaptation is the irreversibility source.

    From Type B genomes, four conditions:
      frozen_cov:  CMA-ES with C forced to I (no covariance learning), sigma0=0.01
      large_sigma: standard CMA-ES with sigma0=1.0 (broad search radius)
      frozen_large: C=I + sigma0=1.0 (isotropic + broad)
      projection:  CANCEL_IDX replaced with Type C (seed 512) values, standard CMA-ES

    Compare against irreversibility test (standard CMA-ES, sigma0=0.01 → 0/15 returned).
    """
    if seed_ids is None:
        seed_ids = [137, 256, 1024]

    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 50

    typec_genome = np.load(os.path.join(PROJECT_DIR, "data", "genome_seed_512.npy"))

    conditions = ["frozen_cov", "large_sigma", "frozen_large", "projection"]
    all_results = {}

    for sid in seed_ids:
        gpath = os.path.join(PROJECT_DIR, "data", f"genome_seed_{sid}.npy")
        if not os.path.exists(gpath):
            log.warning("  genome for seed %d not found, skipping", sid)
            continue

        genome = np.load(gpath)
        baseline = classify_genome(genome, knuth_eval, random_eval,
                                   target_ranks, rel_checker, cfg)
        log.info("=" * 60)
        log.info("COVARIANCE MECHANISM TEST — seed %d (baseline: %s r=%.4f sens=%.3f)",
                 sid, baseline["regime"], baseline["r"],
                 baseline["sensitivity_importance"])
        log.info("=" * 60)

        seed_results = {"baseline": baseline}

        for cond in conditions:
            log.info("  --- condition: %s ---", cond)
            trials = []

            for trial in range(n_trials):
                if cond == "projection":
                    start = genome.copy()
                    start[CANCEL_IDX] = typec_genome[CANCEL_IDX]
                    sigma0 = 0.01
                elif cond == "large_sigma":
                    start = genome.copy()
                    sigma0 = 1.0
                elif cond == "frozen_large":
                    start = genome.copy()
                    sigma0 = 1.0
                else:
                    start = genome.copy()
                    sigma0 = 0.01

                freeze_cov = cond in ("frozen_cov", "frozen_large")

                es = CMAES(GENOME_DIM, sigma0=sigma0, sigma_max=100.0,
                           sigma_min=1e-8, pop_size=pop_size,
                           seed=6000 + sid * 100 + trial)
                es.mean = start.copy()

                pre = classify_genome(start, knuth_eval, random_eval,
                                      target_ranks, rel_checker, cfg)
                best_genome = start.copy()
                best_r = pre["r"]

                for gen in range(refine_gens):
                    try:
                        candidates = es.ask()
                        mu_k = knuth_eval.evaluate_batch(candidates)
                        corr = spearman_batch_gpu(mu_k, target_ranks)
                        bp, bd = check_all_barriers_batch(
                            mu_k, random_eval.evaluate_batch(candidates),
                            candidates, rel_checker,
                            cfg["barriers"], n_vars=cfg["data"]["n_vars"],
                            n_measures=D,
                        )
                    except (RuntimeError, Exception) as e:
                        if "cuda" in str(e).lower():
                            log.warning("    CUDA error: %s", e)
                            break
                        raise

                    nv = np.zeros(pop_size, dtype=np.int32)
                    for j in range(pop_size):
                        if not bp[j]:
                            nv[j] = sum(1 for v in bd[j].values()
                                        if not v["passes"])

                    l2 = np.sqrt((candidates ** 2).sum(axis=1))
                    fit = -corr + penalty * nv + l2_lambda * l2
                    es.tell(candidates, fit)

                    if freeze_cov:
                        es.C = np.eye(es.d)
                        es._stale = True

                    if es.should_restart():
                        es.restart()
                        if freeze_cov:
                            es.C = np.eye(es.d)
                            es._stale = True

                    ib = corr.argmax()
                    if corr[ib] > best_r:
                        best_r = float(corr[ib])
                        best_genome = candidates[ib].copy()

                    if (gen + 1) % 100 == 0:
                        imp = feature_importance(best_genome, D)
                        si_idx = MEASURE_NAMES.index("sensitivity")
                        log.info("    gen %3d | r=%.4f | sens=%.3f | sigma=%.2e",
                                 gen + 1, best_r, float(imp[si_idx]), es.sigma)

                post = classify_genome(best_genome, knuth_eval, random_eval,
                                       target_ranks, rel_checker, cfg)

                trial_result = {
                    "trial": trial, "condition": cond,
                    "pre_r": pre["r"], "pre_regime": pre["regime"],
                    "pre_sens": pre["sensitivity_importance"],
                    "pre_feasible": pre["feasible"],
                    "post_regime": post["regime"], "post_r": post["r"],
                    "post_sens": post["sensitivity_importance"],
                    "post_cross": post["cross_term_frac"],
                    "post_feasible": post["feasible"],
                    "returned_to_c": post["regime"] == "C",
                }
                trials.append(trial_result)
                log.info("  [%s] trial %d: %s->%s r=%.4f sens=%.3f feas=%s %s",
                         cond, trial + 1, pre["regime"], post["regime"],
                         post["r"], post["sensitivity_importance"],
                         post["feasible"],
                         "RETURNED" if trial_result["returned_to_c"] else "STUCK")

            n_returned = sum(1 for t in trials if t["returned_to_c"])
            log.info("  %s: %d/%d returned to Type C", cond, n_returned, n_trials)
            seed_results[cond] = {"trials": trials, "n_returned": n_returned}

        all_results[str(sid)] = seed_results

    out_path = os.path.join(PROJECT_DIR, "data", "covariance_mechanism_test.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out_path)

    log.info("=" * 60)
    log.info("SUMMARY (vs irreversibility test standard: 0/15 returned)")
    log.info("%-14s %s", "Condition",
             "  ".join(f"seed_{s}" for s in seed_ids if str(s) in all_results))
    for cond in conditions:
        counts = []
        for s in seed_ids:
            sk = str(s)
            if sk in all_results and cond in all_results[sk]:
                nr = all_results[sk][cond]["n_returned"]
                counts.append(f"{nr}/{n_trials}")
            else:
                counts.append("N/A")
        log.info("%-14s %s", cond, "      ".join(counts))

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Optimizer comparison — full vs sep vs isotropic CMA-ES
# ═══════════════════════════════════════════════════════════════════════════

def optimizer_comparison(setup_data, seeds=None, max_gen=8000,
                         conditions=None):
    """Compare full/sep/isotropic CMA-ES from scratch.

    Tests whether overhead drift is specific to full covariance adaptation.

    Three conditions:
      full:      standard CMA-ES (full covariance matrix)
      sep:       diagonal-only covariance (off-diag zeroed after update)
      isotropic: C=I frozen throughout (only sigma + mean adapt)

    Tracks regime trajectory at every 1000 gens.
    Saves incrementally after each condition completes.
    """
    if seeds is None:
        seeds = [137, 256, 1024]
    if conditions is None:
        conditions = ["full", "sep", "isotropic"]

    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100

    out_path = os.path.join(PROJECT_DIR, "data", "optimizer_comparison.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        log.info("Loaded existing results: %s", list(all_results.keys()))
    else:
        all_results = {}

    for cond in conditions:
        log.info("=" * 60)
        log.info("OPTIMIZER COMPARISON — %s (seeds=%s, max_gen=%d)",
                 cond, seeds, max_gen)
        log.info("=" * 60)

        cond_results = []

        for seed in seeds:
            log.info("  --- %s seed %d ---", cond, seed)

            es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                       pop_size=pop_size, seed=seed)

            best_genome = None
            best_r = 0.0
            trajectory = []

            for gen in range(max_gen):
                try:
                    candidates = es.ask()
                    mu_k = knuth_eval.evaluate_batch(candidates)
                    corr = spearman_batch_gpu(mu_k, target_ranks)
                    bp, bd = check_all_barriers_batch(
                        mu_k, random_eval.evaluate_batch(candidates),
                        candidates, rel_checker,
                        cfg["barriers"], n_vars=cfg["data"]["n_vars"],
                        n_measures=D,
                    )
                except (RuntimeError, Exception) as e:
                    if "cuda" in str(e).lower():
                        log.warning("  CUDA error at gen %d: %s", gen + 1, e)
                        break
                    raise

                nv = np.zeros(pop_size, dtype=np.int32)
                for j in range(pop_size):
                    if not bp[j]:
                        nv[j] = sum(1 for v in bd[j].values()
                                    if not v["passes"])

                l2 = np.sqrt((candidates ** 2).sum(axis=1))
                fit = -corr + penalty * nv + l2_lambda * l2
                es.tell(candidates, fit)

                if cond == "sep":
                    es.C = np.diag(np.diag(es.C))
                    es._stale = True
                elif cond == "isotropic":
                    es.C = np.eye(es.d)
                    es._stale = True

                if es.should_restart():
                    es.restart()
                    if cond == "sep":
                        es.C = np.diag(np.diag(es.C))
                        es._stale = True
                    elif cond == "isotropic":
                        es.C = np.eye(es.d)
                        es._stale = True

                ib = corr.argmax()
                if corr[ib] > best_r:
                    best_r = float(corr[ib])
                    best_genome = candidates[ib].copy()

                gen_num = gen + 1
                if gen_num % 500 == 0 and best_genome is not None:
                    imp = feature_importance(best_genome, D)
                    si = MEASURE_NAMES.index("sensitivity")
                    log.info("  %s seed %d gen %5d | r=%.4f | sens=%.3f | sigma=%.2e",
                             cond, seed, gen_num, best_r, float(imp[si]),
                             es.sigma)

                if gen_num % 1000 == 0 and best_genome is not None:
                    cls = classify_genome(best_genome, knuth_eval, random_eval,
                                         target_ranks, rel_checker, cfg)
                    trajectory.append({
                        "gen": gen_num,
                        "r": cls["r"], "regime": cls["regime"],
                        "sensitivity_importance": cls["sensitivity_importance"],
                        "cross_term_frac": cls["cross_term_frac"],
                        "feasible": cls["feasible"],
                    })

                if gen_num % 500 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            if best_genome is not None:
                final = classify_genome(best_genome, knuth_eval, random_eval,
                                        target_ranks, rel_checker, cfg)
                cond_results.append({
                    "seed": seed, "final": final, "trajectory": trajectory,
                })
                log.info("  %s seed %d FINAL: r=%.4f regime=%s sens=%.3f "
                         "cross=%.3f feas=%s",
                         cond, seed, final["r"], final["regime"],
                         final["sensitivity_importance"],
                         final["cross_term_frac"], final["feasible"])

        all_results[cond] = cond_results

        regimes = {"A": 0, "B": 0, "C": 0}
        for cr in cond_results:
            regimes[cr["final"]["regime"]] += 1
        log.info("  %s regime distribution: A=%d B=%d C=%d",
                 cond, regimes["A"], regimes["B"], regimes["C"])

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info("  Saved incremental results to %s", out_path)

    # Trajectory summary
    log.info("=" * 60)
    log.info("DRIFT TRAJECTORY (regime at each 1000-gen checkpoint)")
    for cond in conditions:
        for cr in all_results[cond]:
            traj_str = " → ".join(
                f"{t['gen']}:{t['regime']}" for t in cr["trajectory"])
            log.info("  %s seed %d: %s", cond, cr["seed"], traj_str)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Two-stage search — signal-first then barrier-aware
# ═══════════════════════════════════════════════════════════════════════════

def two_stage_search(setup_data, n_seeds=10, stage1_gens=2000, stage2_gens=4000):
    """Two-stage search: signal-first then barrier-aware.

    Stage 1: optimize for r only (no barrier penalty)
    Stage 2: full barrier-aware optimization from stage 1 endpoint
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100
    seed_base = 3000
    all_results = []

    for i in range(n_seeds):
        seed = seed_base + i
        log.info("=" * 50)
        log.info("TWO-STAGE seed %d (%d/%d)", seed, i + 1, n_seeds)

        # Stage 1: signal-only (no barrier penalty)
        log.info("  Stage 1: signal-only (%d gens)", stage1_gens)
        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=seed)
        best_genome = None
        best_r = 0.0

        for gen in range(stage1_gens):
            try:
                candidates = es.ask()
                mu_k = knuth_eval.evaluate_batch(candidates)
                corr = spearman_batch_gpu(mu_k, target_ranks)
            except (RuntimeError, Exception) as e:
                if "cuda" in str(e).lower():
                    log.warning("  CUDA error: %s", e)
                    break
                raise

            l2 = np.sqrt((candidates ** 2).sum(axis=1))
            fitness = -corr + l2_lambda * l2
            es.tell(candidates, fitness)
            if es.should_restart():
                es.restart()

            ib = corr.argmax()
            if corr[ib] > best_r:
                best_r = float(corr[ib])
                best_genome = candidates[ib].copy()

            if (gen + 1) % 500 == 0:
                log.info("    S1 gen %4d | r=%.4f | sigma=%.2e",
                         gen + 1, best_r, es.sigma)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        s1_cls = classify_genome(best_genome, knuth_eval, random_eval,
                                 target_ranks, rel_checker, cfg)
        log.info("  Stage 1 done: r=%.4f feas=%s regime=%s sens=%.3f",
                 s1_cls["r"], s1_cls["feasible"], s1_cls["regime"],
                 s1_cls["sensitivity_importance"])

        # Stage 2: barrier-aware from stage 1 endpoint
        log.info("  Stage 2: barrier-aware (%d gens)", stage2_gens)
        es2 = CMAES(GENOME_DIM, sigma0=0.5, sigma_max=100.0, sigma_min=1e-6,
                    pop_size=pop_size, seed=seed + 10000)
        es2.mean = best_genome.copy()

        for gen in range(stage2_gens):
            try:
                candidates = es2.ask()
                mu_k = knuth_eval.evaluate_batch(candidates)
                corr = spearman_batch_gpu(mu_k, target_ranks)
                bp, bd = check_all_barriers_batch(
                    mu_k, random_eval.evaluate_batch(candidates),
                    candidates, rel_checker,
                    cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                )
            except (RuntimeError, Exception) as e:
                if "cuda" in str(e).lower():
                    log.warning("  CUDA error: %s", e)
                    break
                raise

            nv = np.zeros(pop_size, dtype=np.int32)
            for j in range(pop_size):
                if not bp[j]:
                    nv[j] = sum(1 for v in bd[j].values() if not v["passes"])

            l2 = np.sqrt((candidates ** 2).sum(axis=1))
            fitness = -corr + penalty * nv + l2_lambda * l2
            es2.tell(candidates, fitness)
            if es2.should_restart():
                es2.restart()

            ib = corr.argmax()
            if corr[ib] > best_r:
                best_r = float(corr[ib])
                best_genome = candidates[ib].copy()

            if (gen + 1) % 500 == 0:
                imp = feature_importance(best_genome, D)
                si = MEASURE_NAMES.index("sensitivity")
                log.info("    S2 gen %4d | r=%.4f | sens=%.3f | sigma=%.2e",
                         gen + 1, best_r, float(imp[si]), es2.sigma)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        s2_cls = classify_genome(best_genome, knuth_eval, random_eval,
                                 target_ranks, rel_checker, cfg)
        log.info("  Stage 2 done: r=%.4f feas=%s regime=%s sens=%.3f cross=%.3f",
                 s2_cls["r"], s2_cls["feasible"], s2_cls["regime"],
                 s2_cls["sensitivity_importance"], s2_cls["cross_term_frac"])

        all_results.append({"seed": seed, "stage1": s1_cls, "stage2": s2_cls})

    counts = {"A": 0, "B": 0, "C": 0}
    for r in all_results:
        counts[r["stage2"]["regime"]] += 1
    log.info("\n=== TWO-STAGE RESULTS ===")
    log.info("Final regime distribution: A=%d B=%d C=%d",
             counts["A"], counts["B"], counts["C"])

    out_path = os.path.join(PROJECT_DIR, "data", "two_stage_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_two_stage(all_results)
    return all_results


def _plot_two_stage(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    regime_colors = {"A": "#1976d2", "B": "#ff9800", "C": "#4caf50"}

    # A: transition flow
    ax = axes[0]
    transitions = {}
    for r in all_results:
        key = f'{r["stage1"]["regime"]}->{r["stage2"]["regime"]}'
        transitions[key] = transitions.get(key, 0) + 1
    labels = sorted(transitions.keys())
    counts = [transitions[k] for k in labels]
    ax.barh(range(len(labels)), counts, color='#607d8b', alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title("A. Stage 1 -> Stage 2 transitions")
    ax.grid(True, alpha=0.3, axis='x')

    # B: endpoint scatter (stage 2)
    ax = axes[1]
    for r in all_results:
        s2 = r["stage2"]
        color = regime_colors.get(s2["regime"], '#607d8b')
        marker = 'D' if s2["regime"] == "A" else 's' if s2["regime"] == "B" else '*'
        size = 100 if s2["regime"] in ("A", "C") else 60
        ax.scatter(s2["sensitivity_importance"], s2["cross_term_frac"],
                   c=color, marker=marker, s=size, alpha=0.7,
                   edgecolors='black', linewidth=0.3)
    ax.set_xlabel("Sensitivity importance")
    ax.set_ylabel("Cross-term fraction")
    ax.set_title("B. Two-stage endpoints")
    ax.grid(True, alpha=0.3)

    # C: r comparison
    ax = axes[2]
    s1_rs = [r["stage1"]["r"] for r in all_results]
    s2_rs = [r["stage2"]["r"] for r in all_results]
    ax.scatter(s1_rs, s2_rs, c='#1976d2', s=60, alpha=0.7,
               edgecolors='black', linewidth=0.3)
    ax.plot([0.4, 0.65], [0.4, 0.65], '--', color='gray', alpha=0.5)
    ax.set_xlabel("Stage 1 r (signal-only)")
    ax.set_ylabel("Stage 2 r (barrier-aware)")
    ax.set_title("C. Signal-only vs barrier-aware")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Two-stage search: signal-first then barrier-aware",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", "two_stage_search.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: Regime hit-rate — baseline Type A/B/C distribution
# ═══════════════════════════════════════════════════════════════════════════

def regime_hitrate(setup_data, n_seeds=30, max_gen=4000):
    """Run many seeds with standard search and classify endpoints.

    Provides the baseline Type A/B/C distribution under standard search.
    """
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    pop_size = 100
    seed_base = 4000
    all_results = []

    for i in range(n_seeds):
        seed = seed_base + i
        log.info("=" * 50)
        log.info("HITRATE seed %d (%d/%d)", seed, i + 1, n_seeds)

        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=seed)
        best_genome = None
        best_r = 0.0

        for gen in range(max_gen):
            try:
                candidates = es.ask()
                mu_knuth = knuth_eval.evaluate_batch(candidates)
                corr = spearman_batch_gpu(mu_knuth, target_ranks)
                bp, bd = check_all_barriers_batch(
                    mu_knuth, random_eval.evaluate_batch(candidates),
                    candidates, rel_checker,
                    cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                )
            except (RuntimeError, Exception) as e:
                if "cuda" in str(e).lower():
                    log.warning("  CUDA error: %s", e)
                    break
                raise

            nv = np.zeros(pop_size, dtype=np.int32)
            for j in range(pop_size):
                if not bp[j]:
                    nv[j] = sum(1 for v in bd[j].values() if not v["passes"])

            l2 = np.sqrt((candidates ** 2).sum(axis=1))
            fitness = -corr + penalty * nv + l2_lambda * l2
            es.tell(candidates, fitness)
            if es.should_restart():
                es.restart()

            ib = corr.argmax()
            if corr[ib] > best_r:
                best_r = float(corr[ib])
                best_genome = candidates[ib].copy()

            if (gen + 1) % 1000 == 0:
                imp = feature_importance(best_genome, D)
                si = MEASURE_NAMES.index("sensitivity")
                log.info("    gen %4d | r=%.4f | sens=%.3f | sigma=%.2e",
                         gen + 1, best_r, float(imp[si]), es.sigma)

            if (gen + 1) % 500 == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if best_genome is not None:
            cls = classify_genome(best_genome, knuth_eval, random_eval,
                                  target_ranks, rel_checker, cfg)
            cls["seed"] = seed
            all_results.append(cls)
            log.info("  seed %d: r=%.4f feas=%s regime=%s sens=%.3f cross=%.3f",
                     seed, cls["r"], cls["feasible"], cls["regime"],
                     cls["sensitivity_importance"], cls["cross_term_frac"])

    counts = {"A": 0, "B": 0, "C": 0}
    for r in all_results:
        counts[r["regime"]] += 1
    n = max(len(all_results), 1)
    log.info("\n=== REGIME HIT-RATE ===")
    log.info("Distribution (n=%d): A=%d (%.0f%%) B=%d (%.0f%%) C=%d (%.0f%%)",
             len(all_results),
             counts["A"], 100 * counts["A"] / n,
             counts["B"], 100 * counts["B"] / n,
             counts["C"], 100 * counts["C"] / n)

    out_path = os.path.join(PROJECT_DIR, "data", "regime_hitrate.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out_path)

    _plot_hitrate(all_results)
    return all_results


def _plot_hitrate(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regime_colors = {"A": "#1976d2", "B": "#ff9800", "C": "#4caf50"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: pie chart
    ax = axes[0]
    counts = {"A": 0, "B": 0, "C": 0}
    for r in all_results:
        counts[r["regime"]] += 1
    labels = [f"Type {k}\n({v})" for k, v in counts.items() if v > 0]
    sizes = [v for v in counts.values() if v > 0]
    colors = [regime_colors[k] for k, v in counts.items() if v > 0]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title("A. Regime distribution (standard search)")

    # B: scatter
    ax = axes[1]
    for r in all_results:
        color = regime_colors.get(r["regime"], '#607d8b')
        marker = 'D' if r["regime"] == "A" else 's' if r["regime"] == "B" else '*'
        size = 100 if r["regime"] in ("A", "C") else 60
        ax.scatter(r["sensitivity_importance"], r["cross_term_frac"],
                   c=color, marker=marker, s=size, alpha=0.7,
                   edgecolors='black', linewidth=0.3)
    ax.set_xlabel("Sensitivity importance")
    ax.set_ylabel("Cross-term fraction")
    ax.set_title("B. Endpoint fingerprints")
    ax.grid(True, alpha=0.3)

    # C: r by regime
    ax = axes[2]
    regime_rs = {"A": [], "B": [], "C": []}
    for r in all_results:
        regime_rs[r["regime"]].append(r["r"])
    labels_box, data_box, colors_box = [], [], []
    for k in ["A", "B", "C"]:
        if regime_rs[k]:
            labels_box.append(f"Type {k}")
            data_box.append(regime_rs[k])
            colors_box.append(regime_colors[k])
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors_box[i])
            patch.set_alpha(0.6)
    ax.set_ylabel("Final Spearman r")
    ax.set_title("C. Prediction by regime")
    ax.grid(True, alpha=0.3)

    n = len(all_results)
    fig.suptitle(f"Regime hit-rate: standard search (n={n})",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PROJECT_DIR, "figures", "regime_hitrate.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fig_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]

    if exp == "plot":
        plot_dynamics()
        return

    log.info("Setting up evaluators...")
    setup_data = setup()

    if exp == "run":
        max_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        run_dynamics(setup_data, max_gen)
        plot_dynamics()

    elif exp == "warm_start":
        results = load_results()
        ws = warm_start_experiment(setup_data)
        results["warm_start"] = ws
        save_results(results)

    elif exp == "truncation":
        progressive_truncation(setup_data)

    elif exp == "truncation_seed":
        if len(sys.argv) < 3:
            log.error("Usage: dynamics.py truncation_seed <seed_id>")
            sys.exit(1)
        seed_id = int(sys.argv[2])
        truncation_seed(setup_data, seed_id)

    elif exp == "multiseed":
        max_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        seeds = [int(s) for s in sys.argv[3:]] if len(sys.argv) > 3 else None
        multiseed_dynamics(setup_data, max_gen=max_gen, seeds=seeds)

    elif exp == "endpoint":
        if len(sys.argv) < 3:
            log.error("Usage: dynamics.py endpoint <seed_id> [seed_id ...]")
            sys.exit(1)
        for sid in sys.argv[2:]:
            endpoint_diagnostics(setup_data, int(sid))

    elif exp == "interpolate":
        if len(sys.argv) < 4:
            log.error("Usage: dynamics.py interpolate <seed_a> <seed_b> [n_steps]")
            sys.exit(1)
        sa, sb = int(sys.argv[2]), int(sys.argv[3])
        ns = int(sys.argv[4]) if len(sys.argv) > 4 else 21
        interpolate_genomes(setup_data, sa, sb, ns)

    elif exp == "basin_volume":
        if len(sys.argv) < 3:
            log.error("Usage: dynamics.py basin_volume <seed_id> [n_samples]")
            sys.exit(1)
        sid = int(sys.argv[2])
        ns = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        basin_volume(setup_data, sid, ns)

    elif exp == "injection":
        if len(sys.argv) < 3:
            log.error("Usage: dynamics.py injection <seed_id>")
            sys.exit(1)
        sid = int(sys.argv[2])
        sensitivity_injection(setup_data, sid)

    elif exp == "local_cont":
        sid = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        nps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        rg = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        local_continuation(setup_data, seed_id=sid, n_per_sigma=nps, refine_gens=rg)

    elif exp == "targeted":
        ns = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        mg = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        lo = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
        targeted_recovery(setup_data, n_seeds=ns, max_gen=mg, lambda_oh=lo)

    elif exp == "two_stage":
        ns = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        s1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        s2 = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
        two_stage_search(setup_data, n_seeds=ns, stage1_gens=s1, stage2_gens=s2)

    elif exp == "hitrate":
        ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        mg = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        regime_hitrate(setup_data, n_seeds=ns, max_gen=mg)

    elif exp == "irreversibility":
        sids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
        irreversibility_test(setup_data, seed_ids=sids)

    elif exp == "ablation":
        sid = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        sensitivity_ablation(setup_data, seed_id=sid)

    elif exp == "cov_mech":
        sids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
        covariance_mechanism_test(setup_data, seed_ids=sids)

    elif exp == "opt_compare":
        mg = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        rest = sys.argv[3:]
        sds = None
        conds = None
        for i, a in enumerate(rest):
            if a == "--conds":
                conds = rest[i + 1].split(",")
                rest = rest[:i] + rest[i + 2:]
                break
        if rest:
            sds = [int(x) for x in rest]
        optimizer_comparison(setup_data, seeds=sds, max_gen=mg,
                             conditions=conds)

    else:
        log.error("Unknown: %s", exp)
        sys.exit(1)


if __name__ == "__main__":
    main()
