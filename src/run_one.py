"""Run one experiment at a time. Usage: python run_one.py <experiment_name>

Experiments:
  family_threshold_only, family_full — remaining B families
  lock_spectral_search_sensitivity, lock_sensitivity_search_spectral,
  lock_spectral_search_rest, lock_sensitivity_search_rest — axis-locking (C)
"""

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
    load_data, load_best_genome, MEASURE_NAMES, D, Q_DIM,
    Q_SLICE, W_SLICE, A_SLICE, T_SLICE, GENOME_DIM,
)
from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
from barriers import check_all_barriers_batch, RelativizationChecker
from search import CMAES


def setup_evaluators():
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
    return measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg


def run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
              locked_genome=None, locked_idx=None, free_idx=None,
              family_mask=None, max_gen=200, pop_size=100):
    """Generic CMA-ES runner with optional locking or family masking."""
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]

    es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=pop_size, seed=42)

    best_r = 0.0
    best_r_feasible = 0.0
    convergence = []
    stall_count = 0
    prev_best = 0.0

    for gen in range(max_gen):
        candidates = es.ask()

        # family mask
        if family_mask:
            if not family_mask.get("w"):
                candidates[:, W_SLICE] = 0.0
            if not family_mask.get("q"):
                candidates[:, Q_SLICE] = 0.0
            if not family_mask.get("a"):
                candidates[:, A_SLICE] = 0.0
                candidates[:, T_SLICE] = 0.0

        # axis locking
        if locked_genome is not None and locked_idx is not None:
            for idx in locked_idx:
                candidates[:, idx] = locked_genome[idx]
            if free_idx is not None:
                for idx in range(GENOME_DIM):
                    if idx not in locked_idx and idx not in free_idx:
                        candidates[:, idx] = 0.0

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

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall_count = 0
        else:
            stall_count += 1

        if gen % 50 == 0 or gen == max_gen - 1:
            convergence.append({
                "gen": gen + 1, "best_r": best_r,
                "best_r_feasible": best_r_feasible,
                "n_feasible": int(barrier_pass.sum()),
            })
            log.info("  gen %4d | best_r %.4f  feasible %.4f | %d/%d | sigma %.2e",
                     gen + 1, best_r, best_r_feasible,
                     int(barrier_pass.sum()), pop_size, es.sigma)

        if stall_count >= 100:
            log.info("  Early stop at gen %d", gen + 1)
            break

    return {"best_r": best_r, "best_r_feasible": best_r_feasible,
            "convergence": convergence}


def get_feature_indices(fi):
    indices = [fi]
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == fi or j == fi:
                indices.append(D + k)
            k += 1
    indices.append(D + Q_DIM + fi)
    indices.append(D + Q_DIM + D + fi)
    return set(indices)


def save_result(key, value):
    result_path = os.path.join(PROJECT_DIR, "data", "mechanism_results.json")
    with open(result_path) as f:
        data = json.load(f)

    # navigate to right spot
    parts = key.split(".")
    target = data
    for p in parts[:-1]:
        if p not in target:
            target[p] = {}
        target = target[p]
    target[parts[-1]] = value

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved %s to mechanism_results.json", key)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]
    log.info("=== Running experiment: %s ===", exp)
    t0 = time.time()

    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg = setup_evaluators()
    genome = load_best_genome()

    si = MEASURE_NAMES.index("sensitivity")
    se = MEASURE_NAMES.index("spectral_entropy")
    sens_idx = get_feature_indices(si)
    spec_idx = get_feature_indices(se)

    if exp == "family_threshold_only":
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      family_mask={"w": False, "q": False, "a": True})
        save_result("family_stress_test.threshold_only", r)

    elif exp == "family_full":
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      family_mask={"w": True, "q": True, "a": True})
        save_result("family_stress_test.full", r)

    elif exp == "lock_spectral_search_sensitivity":
        free = sens_idx - spec_idx
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      locked_genome=genome, locked_idx=list(spec_idx),
                      free_idx=list(free))
        if "axis_locking" not in json.load(open(os.path.join(PROJECT_DIR, "data", "mechanism_results.json"))):
            save_result("axis_locking", {})
        save_result("axis_locking.lock_spectral_search_sensitivity", r)

    elif exp == "lock_sensitivity_search_spectral":
        free = spec_idx - sens_idx
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      locked_genome=genome, locked_idx=list(sens_idx),
                      free_idx=list(free))
        save_result("axis_locking.lock_sensitivity_search_spectral", r)

    elif exp == "lock_spectral_search_rest":
        free = set(range(GENOME_DIM)) - spec_idx
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      locked_genome=genome, locked_idx=list(spec_idx),
                      free_idx=list(free))
        save_result("axis_locking.lock_spectral_search_rest", r)

    elif exp == "lock_sensitivity_search_rest":
        free = set(range(GENOME_DIM)) - sens_idx
        r = run_cmaes(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                      locked_genome=genome, locked_idx=list(sens_idx),
                      free_idx=list(free))
        save_result("axis_locking.lock_sensitivity_search_rest", r)

    else:
        log.error("Unknown experiment: %s", exp)
        sys.exit(1)

    log.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
