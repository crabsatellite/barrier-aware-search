"""Finish Package B (threshold_only + full families) then run Package C (axis-locking)."""

import json
import logging
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "mechanism.log")),
    ],
)
log = logging.getLogger(__name__)

from mechanism import (
    load_data, load_best_genome, evaluate_genome,
    MEASURE_NAMES, D, Q_DIM, Q_SLICE, W_SLICE, A_SLICE, T_SLICE, GENOME_DIM,
    axis_locking_test,
)
from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
from barriers import check_all_barriers_batch, RelativizationChecker
from search import CMAES


def run_family(family_name, mask, knuth_eval, random_eval, target_ranks,
               rel_checker, cfg, pop_size=100, max_gen=500):
    """Run one CMA-ES family search."""
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]

    log.info("  --- Family: %s ---", family_name)
    t0 = time.time()

    es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=pop_size, seed=42)

    best_r = 0.0
    best_r_feasible = 0.0
    convergence = []
    n_feasible_best = 0
    stall_count = 0
    prev_best = 0.0

    for gen in range(max_gen):
        candidates = es.ask()

        if not mask["w"]:
            candidates[:, W_SLICE] = 0.0
        if not mask["q"]:
            candidates[:, Q_SLICE] = 0.0
        if not mask["a"]:
            candidates[:, A_SLICE] = 0.0
            candidates[:, T_SLICE] = 0.0

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

        gen_best = float(corr.max())
        feasible_corrs = corr[barrier_pass.astype(bool)]
        gen_best_feas = float(feasible_corrs.max()) if len(feasible_corrs) > 0 else 0.0

        if gen_best > best_r:
            best_r = gen_best
        if gen_best_feas > best_r_feasible:
            best_r_feasible = gen_best_feas
            n_feasible_best = int(barrier_pass.sum())

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall_count = 0
        else:
            stall_count += 1

        if gen % 100 == 0 or gen == max_gen - 1:
            convergence.append({
                "gen": gen + 1, "best_r": best_r,
                "best_r_feasible": best_r_feasible,
                "n_feasible": int(barrier_pass.sum()),
            })
            log.info("    gen %4d | best_r %.4f  feasible_best %.4f | feasible %d/%d | sigma %.2e",
                     gen + 1, best_r, best_r_feasible,
                     int(barrier_pass.sum()), pop_size, es.sigma)

        if stall_count >= 200:
            log.info("    Early stop: no improvement for 200 gens")
            convergence.append({
                "gen": gen + 1, "best_r": best_r,
                "best_r_feasible": best_r_feasible,
                "n_feasible": int(barrier_pass.sum()),
            })
            break

    elapsed = time.time() - t0
    result = {
        "best_r": best_r,
        "best_r_feasible": best_r_feasible,
        "n_feasible": n_feasible_best,
        "elapsed_s": elapsed,
        "convergence": convergence,
    }
    log.info("  %s: best_r=%.4f  feasible_best=%.4f  (%.0fs)",
             family_name, best_r, best_r_feasible, elapsed)
    return result


def main():
    t_start = time.time()
    measures, quad, targets = load_data()
    genome = load_best_genome()

    log.info("=" * 70)
    log.info("FINISHING PACKAGE B + RUNNING PACKAGE C")
    log.info("=" * 70)

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

    # Load existing results
    result_path = os.path.join(PROJECT_DIR, "data", "mechanism_results.json")
    with open(result_path) as f:
        existing = json.load(f)

    # B: Run remaining families
    remaining = {
        "threshold_only": {"w": False, "q": False, "a": True},
        "full": {"w": True, "q": True, "a": True},
    }

    family_results = existing.get("family_stress_test") or {}
    for name, mask in remaining.items():
        family_results[name] = run_family(
            name, mask, knuth_eval, random_eval, target_ranks, rel_checker, cfg)

    # Also fill in earlier results from log
    if "linear_only" not in family_results:
        family_results["linear_only"] = {
            "best_r": 0.5148, "best_r_feasible": 0.0, "n_feasible": 0,
            "elapsed_s": 729, "convergence": [],
        }
    if "linear_quad" not in family_results:
        family_results["linear_quad"] = {
            "best_r": 0.5828, "best_r_feasible": 0.0, "n_feasible": 0,
            "elapsed_s": 800, "convergence": [],
        }
    if "linear_threshold" not in family_results:
        family_results["linear_threshold"] = {
            "best_r": 0.5461, "best_r_feasible": 0.4997, "n_feasible": 0,
            "elapsed_s": 825, "convergence": [],
        }
    if "quad_only" not in family_results:
        family_results["quad_only"] = {
            "best_r": 0.4434, "best_r_feasible": 0.0, "n_feasible": 0,
            "elapsed_s": 818, "convergence": [],
        }

    # Summary
    log.info("")
    log.info("=== Family comparison (complete) ===")
    log.info("  %-20s  %8s  %8s", "Family", "Best r", "Feas. r")
    for name in ["linear_only", "linear_quad", "linear_threshold",
                 "quad_only", "threshold_only", "full"]:
        if name in family_results:
            r = family_results[name]
            log.info("  %-20s  %8.4f  %8.4f", name, r["best_r"], r["best_r_feasible"])

    existing["family_stress_test"] = family_results

    # Save intermediate
    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    log.info("Family results saved.")

    # C: Axis-locking
    axis_lock = axis_locking_test(genome, measures, quad, targets)
    existing["axis_locking"] = axis_lock

    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    # Generate figures
    log.info("")
    log.info("Generating figures...")
    try:
        from mechanism import generate_figures
        generate_figures(existing["ablation"], existing["interaction"],
                         existing["threshold_activation"], existing["family_stress_test"],
                         genome, measures, targets)
    except Exception as e:
        log.warning("Figure generation failed: %s", e)

    total = time.time() - t_start
    log.info("Total elapsed: %.1f min", total / 60)


if __name__ == "__main__":
    main()
