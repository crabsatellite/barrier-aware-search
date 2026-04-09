"""
Robustness validation: does the signal/calibration decomposition survive structural perturbation?

Experiments:
  barrier_ablation  — no/single/dual/triple barrier, compare feature decomposition
  alt_searchers     — random search + DE, check if same attractor emerges
  barrier_sweep     — vary barrier thresholds, observe phase transitions

Usage: python robustness.py <experiment> [sub-experiment]
  python robustness.py barrier_ablation none
  python robustness.py barrier_ablation natural_only
  python robustness.py barrier_ablation relativ_only
  python robustness.py barrier_ablation algebrize_only
  python robustness.py alt_searchers random
  python robustness.py alt_searchers de
  python robustness.py analyze          — compare all completed runs
"""

import json
import logging
import os
import sys
import time
import copy

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
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "robustness.log")),
    ],
)
log = logging.getLogger(__name__)

from mechanism import (
    MEASURE_NAMES, D, Q_DIM, GENOME_DIM, W_SLICE, Q_SLICE, A_SLICE, T_SLICE,
    load_data,
)
from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu, genome_dim
from barriers import check_all_barriers_batch, RelativizationChecker
from search import CMAES
from analyze_candidates import feature_importance


RESULT_PATH = os.path.join(PROJECT_DIR, "data", "robustness_results.json")


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


def evaluate_candidates(candidates, knuth_eval, random_eval, target_ranks,
                        rel_checker, cfg, barrier_mask=None, l2_lambda=0.001):
    """Evaluate with optional barrier masking."""
    pop = len(candidates)
    mu_knuth = knuth_eval.evaluate_batch(candidates)
    mu_random = random_eval.evaluate_batch(candidates)
    corr = spearman_batch_gpu(mu_knuth, target_ranks)

    barrier_pass, barrier_details = check_all_barriers_batch(
        mu_knuth, mu_random, candidates, rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )

    # apply barrier mask: if a barrier is disabled, treat it as always passing
    if barrier_mask:
        for i in range(pop):
            for bname, enabled in barrier_mask.items():
                if not enabled and bname in barrier_details[i]:
                    barrier_details[i][bname]["passes"] = True
            barrier_pass[i] = all(v["passes"] for v in barrier_details[i].values())

    n_violations = np.zeros(pop, dtype=np.int32)
    for i in range(pop):
        if not barrier_pass[i]:
            n_violations[i] = sum(1 for v in barrier_details[i].values()
                                  if not v["passes"])

    penalty = cfg["search"]["barrier_penalty"]
    l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
    fitness = -corr + penalty * n_violations + l2_lambda * l2_norm

    return fitness, corr, barrier_pass, barrier_details


def run_cmaes_search(knuth_eval, random_eval, target_ranks, rel_checker, cfg,
                     barrier_mask=None, max_gen=200, pop_size=100, seed=42):
    """Run CMA-ES search with optional barrier masking."""
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)

    es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=pop_size, seed=seed)

    best_r = 0.0
    best_genome = None
    convergence = []
    stall_count = 0
    prev_best = 0.0

    for gen in range(max_gen):
        candidates = es.ask()
        fitness, corr, bp, bd = evaluate_candidates(
            candidates, knuth_eval, random_eval, target_ranks,
            rel_checker, cfg, barrier_mask, l2_lambda)
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

        if gen % 50 == 0 or gen == max_gen - 1:
            feas_corrs = corr[bp.astype(bool)]
            feas_best = float(feas_corrs.max()) if len(feas_corrs) > 0 else 0.0
            convergence.append({
                "gen": gen + 1, "best_r": best_r, "feas_r": feas_best,
                "n_feasible": int(bp.sum()),
            })
            log.info("  gen %4d | best_r %.4f  feas %.4f | %d/%d | sigma %.2e",
                     gen + 1, best_r, feas_best, int(bp.sum()), pop_size, es.sigma)

        if stall_count >= 100:
            log.info("  Early stop at gen %d", gen + 1)
            break

    return best_genome, best_r, convergence


def analyze_genome(genome, measures, quad, targets):
    """Quick ablation + feature importance for a genome."""
    from mechanism import evaluate_genome

    full_r = evaluate_genome(genome, measures, quad, targets)
    importance = feature_importance(genome, D)

    # top-3 ablation
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
        "importance": {MEASURE_NAMES[i]: float(importance[i]) for i in range(D)},
        "ablation_drops": drops,
        "top_signal": sorted(drops.items(), key=lambda x: -x[1])[:3],
        "top_importance": sorted(
            [(MEASURE_NAMES[i], float(importance[i])) for i in range(D)],
            key=lambda x: -x[1])[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Barrier ablation
# ═══════════════════════════════════════════════════════════════════════════

BARRIER_CONFIGS = {
    "none":           {"natural_proof": False, "relativization": False, "algebrization": False},
    "natural_only":   {"natural_proof": True,  "relativization": False, "algebrization": False},
    "relativ_only":   {"natural_proof": False, "relativization": True,  "algebrization": False},
    "algebrize_only": {"natural_proof": False, "relativization": False, "algebrization": True},
    "nat_relativ":    {"natural_proof": True,  "relativization": True,  "algebrization": False},
    "nat_algebrize":  {"natural_proof": True,  "relativization": False, "algebrization": True},
    "relativ_algebrize": {"natural_proof": False, "relativization": True, "algebrization": True},
    "all_three":      {"natural_proof": True,  "relativization": True,  "algebrization": True},
}


def run_barrier_ablation(config_name, setup_data):
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    barrier_mask = BARRIER_CONFIGS[config_name]
    active = [k for k, v in barrier_mask.items() if v]
    log.info("=== Barrier ablation: %s (active: %s) ===",
             config_name, active or "NONE")

    t0 = time.time()
    genome, best_r, convergence = run_cmaes_search(
        knuth_eval, random_eval, target_ranks, rel_checker, cfg,
        barrier_mask=barrier_mask, max_gen=120)

    analysis = analyze_genome(genome, measures, quad, targets)
    elapsed = time.time() - t0

    result = {
        "config": config_name,
        "barrier_mask": barrier_mask,
        "best_r_gpu": best_r,
        "analysis": analysis,
        "convergence": convergence,
        "elapsed_s": elapsed,
        "genome": genome.tolist(),
    }

    log.info("  Full r (scipy) = %.4f", analysis["full_r"])
    log.info("  Top signal features: %s",
             ", ".join(f"{n}={d:.4f}" for n, d in analysis["top_signal"]))
    log.info("  Top importance: %s",
             ", ".join(f"{n}={w:.3f}" for n, w in analysis["top_importance"]))
    log.info("  Done in %.1f min", elapsed / 60)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Alternative searchers
# ═══════════════════════════════════════════════════════════════════════════

def run_random_search(setup_data, n_eval=20000, elite_size=50):
    """Pure random search with elite selection."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    log.info("=== Random search (%d evaluations, elite %d) ===", n_eval, elite_size)
    t0 = time.time()
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    rng = np.random.RandomState(42)

    best_r = 0.0
    best_genome = None
    convergence = []
    batch_size = 100

    for batch in range(n_eval // batch_size):
        candidates = rng.randn(batch_size, GENOME_DIM).astype(np.float32)
        fitness, corr, bp, bd = evaluate_candidates(
            candidates, knuth_eval, random_eval, target_ranks,
            rel_checker, cfg, l2_lambda=l2_lambda)

        idx = corr.argmax()
        if corr[idx] > best_r:
            best_r = float(corr[idx])
            best_genome = candidates[idx].copy()

        if (batch + 1) % 20 == 0 or batch == 0:
            n_eval_so_far = (batch + 1) * batch_size
            convergence.append({"n_eval": n_eval_so_far, "best_r": best_r})
            log.info("  %5d evals | best_r %.4f", n_eval_so_far, best_r)

    analysis = analyze_genome(best_genome, measures, quad, targets)
    elapsed = time.time() - t0

    result = {
        "method": "random_search",
        "n_evaluations": n_eval,
        "best_r_gpu": best_r,
        "analysis": analysis,
        "convergence": convergence,
        "elapsed_s": elapsed,
        "genome": best_genome.tolist(),
    }

    log.info("  Full r (scipy) = %.4f", analysis["full_r"])
    log.info("  Top signal: %s",
             ", ".join(f"{n}={d:.4f}" for n, d in analysis["top_signal"]))
    log.info("  Done in %.1f min", elapsed / 60)
    return result


def run_differential_evolution(setup_data, max_gen=200, pop_size=100):
    """Simple DE/rand/1/bin."""
    measures, quad, targets, knuth_eval, random_eval, target_ranks, rel_checker, cfg, _ = setup_data

    log.info("=== Differential Evolution (pop=%d, gen=%d) ===", pop_size, max_gen)
    t0 = time.time()
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    rng = np.random.RandomState(42)

    F = 0.8   # mutation factor
    CR = 0.9  # crossover rate

    # initialize population
    pop = rng.randn(pop_size, GENOME_DIM).astype(np.float32)
    fitness, corr, bp, bd = evaluate_candidates(
        pop, knuth_eval, random_eval, target_ranks,
        rel_checker, cfg, l2_lambda=l2_lambda)
    pop_fitness = fitness.copy()

    best_r = float(corr.max())
    best_genome = pop[corr.argmax()].copy()
    convergence = []
    stall_count = 0
    prev_best = 0.0

    for gen in range(max_gen):
        # generate trial vectors
        trials = np.zeros_like(pop)
        for i in range(pop_size):
            # select 3 distinct random indices != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = rng.choice(idxs, 3, replace=False)

            # mutation: DE/rand/1
            mutant = pop[a] + F * (pop[b] - pop[c])

            # binomial crossover
            j_rand = rng.randint(GENOME_DIM)
            mask = rng.rand(GENOME_DIM) < CR
            mask[j_rand] = True
            trials[i] = np.where(mask, mutant, pop[i])

        # evaluate trials
        trial_fitness, trial_corr, trial_bp, trial_bd = evaluate_candidates(
            trials, knuth_eval, random_eval, target_ranks,
            rel_checker, cfg, l2_lambda=l2_lambda)

        # selection (minimize fitness)
        improved = trial_fitness < pop_fitness
        pop[improved] = trials[improved]
        pop_fitness[improved] = trial_fitness[improved]

        # track best
        best_idx = trial_corr.argmax()
        if trial_corr[best_idx] > best_r:
            best_r = float(trial_corr[best_idx])
            best_genome = trials[best_idx].copy()

        if best_r > prev_best + 1e-5:
            prev_best = best_r
            stall_count = 0
        else:
            stall_count += 1

        if gen % 50 == 0 or gen == max_gen - 1:
            convergence.append({"gen": gen + 1, "best_r": best_r})
            log.info("  gen %4d | best_r %.4f | improved %d/%d",
                     gen + 1, best_r, int(improved.sum()), pop_size)

        if stall_count >= 100:
            log.info("  Early stop at gen %d", gen + 1)
            break

    analysis = analyze_genome(best_genome, measures, quad, targets)
    elapsed = time.time() - t0

    result = {
        "method": "differential_evolution",
        "best_r_gpu": best_r,
        "analysis": analysis,
        "convergence": convergence,
        "elapsed_s": elapsed,
        "genome": best_genome.tolist(),
    }

    log.info("  Full r (scipy) = %.4f", analysis["full_r"])
    log.info("  Top signal: %s",
             ", ".join(f"{n}={d:.4f}" for n, d in analysis["top_signal"]))
    log.info("  Done in %.1f min", elapsed / 60)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Cross-run analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_all():
    """Compare feature decompositions across all robustness runs."""
    log.info("=" * 70)
    log.info("CROSS-RUN COMPARISON")
    log.info("=" * 70)

    results = load_results()

    # barrier ablation comparison
    if "barrier_ablation" in results:
        log.info("")
        log.info("--- Barrier Ablation: Feature Decomposition ---")
        log.info("  %-20s  %6s  %s", "Config", "r", "Top-3 signal features (ablation drop)")

        for name, r in sorted(results["barrier_ablation"].items(),
                               key=lambda x: -x[1].get("analysis", {}).get("full_r", 0)):
            analysis = r.get("analysis", {})
            full_r = analysis.get("full_r", 0)
            top = analysis.get("top_signal", [])
            top_str = ", ".join(f"{n}={d:.3f}" for n, d in top[:3])
            log.info("  %-20s  %.4f  %s", name, full_r, top_str)

        # check if spectral_entropy is always #1 signal
        log.info("")
        log.info("  --- Signal axis stability ---")
        for name, r in results["barrier_ablation"].items():
            top = r.get("analysis", {}).get("top_signal", [])
            if top:
                log.info("    %-20s  #1 signal: %-22s (drop=%.4f)",
                         name, top[0][0], top[0][1])

        # check if sensitivity importance is always high
        log.info("")
        log.info("  --- Sensitivity weight share ---")
        for name, r in results["barrier_ablation"].items():
            imp = r.get("analysis", {}).get("importance", {})
            sens = imp.get("sensitivity", 0)
            log.info("    %-20s  sensitivity importance: %.3f", name, sens)

    # alternative searchers
    if "alt_searchers" in results:
        log.info("")
        log.info("--- Alternative Searchers ---")
        for name, r in results["alt_searchers"].items():
            analysis = r.get("analysis", {})
            full_r = analysis.get("full_r", 0)
            top = analysis.get("top_signal", [])
            top_str = ", ".join(f"{n}={d:.3f}" for n, d in top[:3])
            log.info("  %-20s  r=%.4f  %s", name, full_r, top_str)

    # save comparison summary
    results["_comparison_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(results)
    log.info("Comparison complete.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else None

    if exp == "analyze":
        analyze_all()
        return

    log.info("Setting up evaluators...")
    setup_data = setup()
    results = load_results()

    if exp == "barrier_ablation":
        if not sub or sub not in BARRIER_CONFIGS:
            log.error("Specify config: %s", list(BARRIER_CONFIGS.keys()))
            sys.exit(1)
        result = run_barrier_ablation(sub, setup_data)
        if "barrier_ablation" not in results:
            results["barrier_ablation"] = {}
        results["barrier_ablation"][sub] = result
        save_results(results)

    elif exp == "alt_searchers":
        if sub == "random":
            result = run_random_search(setup_data)
        elif sub == "de":
            result = run_differential_evolution(setup_data)
        else:
            log.error("Specify: random or de")
            sys.exit(1)
        if "alt_searchers" not in results:
            results["alt_searchers"] = {}
        results["alt_searchers"][sub] = result
        save_results(results)

    else:
        log.error("Unknown experiment: %s", exp)
        sys.exit(1)


if __name__ == "__main__":
    main()
