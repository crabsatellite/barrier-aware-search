"""
Block 5A: Default lambda replication (lambda=17 vs lambda=100).
Block 6A: Extended escape attempts (8000-gen, with mutation acceptance diagnostics).

Usage:
    python exp_lambda_escape.py lambda        # Block 5A
    python exp_lambda_escape.py escape        # Block 6A
    python exp_lambda_escape.py all
    python exp_lambda_escape.py status
"""

import argparse
import os
import sys
import time

import numpy as np

from shared import (
    get_logger, RESULTS_DIR, load_results, save_results,
    result_key, setup_boolean_domain,
)

log = get_logger("exp_lambda_escape")

LAMBDA_PATH = os.path.join(RESULTS_DIR, "lambda_comparison.json")
ESCAPE_PATH = os.path.join(RESULTS_DIR, "extended_escape.json")

LAMBDA_SEEDS = list(range(1, 31))
VARIANTS = ["full", "sep", "isotropic"]
LAMBDAS = [17, 50, 100, 200]
CHECKPOINTS = [500, 1000, 2000, 3000]


# ═══════════════════════════════════════════════════════════════════════════
# Block 5A: Lambda sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_lambda():
    """Test drift at different population sizes."""
    log.info("=" * 70)
    log.info("Block 5A: Lambda sweep comparison")
    log.info("=" * 70)

    results = load_results(LAMBDA_PATH)
    domain = setup_boolean_domain()

    # focus on full vs sep at each lambda level, 10 seeds each for efficiency
    focus_seeds = list(range(1, 11))
    focus_variants = ["full", "sep"]

    for lam in LAMBDAS:
        log.info("\n--- Lambda = %d ---", lam)
        lam_key = f"lambda_{lam}"
        if lam_key not in results:
            results[lam_key] = {}

        for seed in focus_seeds:
            for variant in focus_variants:
                key = result_key(variant, seed)
                if key in results[lam_key]:
                    log.info("  [skip] %s (lam=%d)", key, lam)
                    continue

                log.info("  [run]  %s (lam=%d) ...", key, lam)

                from shared import run_cmaes_trial
                trial = run_cmaes_trial(
                    seed=seed, variant=variant,
                    max_gen=3000, pop_size=lam,
                    domain=domain, checkpoints=CHECKPOINTS,
                )
                results[lam_key][key] = trial
                save_results(results, LAMBDA_PATH)
                log.info("  [done] %s — regime=%s (%.1fs)",
                         key, trial["final"]["regime"], trial["elapsed_s"])

    # summary
    log.info("\nLambda sweep summary:")
    for lam in LAMBDAS:
        lam_key = f"lambda_{lam}"
        if lam_key not in results:
            continue
        for variant in focus_variants:
            drift = sum(1 for s in focus_seeds
                        if result_key(variant, s) in results[lam_key]
                        and results[lam_key][result_key(variant, s)]
                        ["final"]["regime"] in ("A", "B"))
            total = sum(1 for s in focus_seeds
                        if result_key(variant, s) in results[lam_key])
            if total:
                log.info("  lam=%-4d %-8s: drift %d/%d", lam, variant, drift, total)

    save_results(results, LAMBDA_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# Block 6A: Extended escape attempts with diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def run_escape():
    """Full-budget escape from Type B endpoints with mutation acceptance tracking."""
    log.info("=" * 70)
    log.info("Block 6A: Extended escape attempts (8000-gen)")
    log.info("=" * 70)

    results = load_results(ESCAPE_PATH)
    domain = setup_boolean_domain()

    # Load existing Type B endpoints from the 30-seed experiment
    crossopt_path = os.path.join(RESULTS_DIR, "crossopt_30seed.json")
    crossopt = load_results(crossopt_path)

    # find all Type B endpoints from full CMA-ES
    type_b_seeds = []
    for seed in range(1, 31):
        key = result_key("full", seed)
        if key in crossopt:
            regime = crossopt[key].get("final", {}).get("regime", "")
            if regime in ("A", "B"):
                genome = crossopt[key].get("genome")
                if genome is not None:
                    type_b_seeds.append((seed, np.array(genome)))

    if not type_b_seeds:
        log.warning("No Type B endpoints found. Run exp_30seed.py crossopt first.")
        # fallback: use original seeds from data/
        for seed_id in [137, 256, 1024]:
            ep_path = os.path.join(
                os.path.dirname(RESULTS_DIR), f"endpoint_seed_{seed_id}.json")
            if os.path.exists(ep_path):
                import json
                with open(ep_path) as f:
                    ep = json.load(f)
                # try to load genome from multiseed_results
                ms_path = os.path.join(
                    os.path.dirname(RESULTS_DIR), "multiseed_results.json")
                if os.path.exists(ms_path):
                    with open(ms_path) as f:
                        ms = json.load(f)
                    for entry in ms.get("seeds", []):
                        if entry.get("seed") == seed_id:
                            genome = entry.get("genome")
                            if genome:
                                type_b_seeds.append((seed_id, np.array(genome)))
                                break

    log.info("Found %d Type B/A endpoints for escape testing", len(type_b_seeds))

    escape_conditions = [
        ("full_standard", "full", 0.3),
        ("full_large_sigma", "full", 1.0),
        ("sep_from_B", "sep", 0.3),
        ("frozen_cov", "isotropic", 0.3),
    ]

    from search import CMAES
    from measures import spearman_batch_gpu
    from barriers import check_all_barriers_batch
    from analyze_candidates import feature_importance

    D = domain["D"]
    GENOME_DIM = domain["GENOME_DIM"]
    knuth_eval = domain["knuth_eval"]
    random_eval = domain["random_eval"]
    target_ranks = domain["target_ranks"]
    rel_checker = domain["rel_checker"]
    cfg = domain["cfg"]
    penalty_val = cfg["search"]["barrier_penalty"]
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    si = 8

    for seed_id, start_genome in type_b_seeds[:5]:  # cap at 5 endpoints
        for cond_name, variant, sigma0 in escape_conditions:
            key = f"seed{seed_id}_{cond_name}"
            if key in results:
                log.info("  [skip] %s", key)
                continue

            log.info("  [run]  %s ...", key)

            es = CMAES(GENOME_DIM, sigma0=sigma0, sigma_max=100.0, sigma_min=1e-6,
                       pop_size=100, seed=seed_id + 9999)
            es.mean = start_genome.copy()

            # pre-escape classification
            imp_pre = feature_importance(start_genome, D)
            phi_pre = float(imp_pre[si])

            trajectory = []
            mutations_accepted = 0
            total_gens = 0
            best_fitness_prev = None
            t0 = time.time()

            for gen in range(3000):
                candidates = es.ask()
                mu_k = knuth_eval.evaluate_batch(candidates)
                mu_r = random_eval.evaluate_batch(candidates)
                corr = spearman_batch_gpu(mu_k, target_ranks)
                bp, bd = check_all_barriers_batch(
                    mu_k, mu_r, candidates, rel_checker,
                    cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
                )
                n_viol = np.array([
                    sum(1 for v in bd[i].values() if not v["passes"])
                    for i in range(len(candidates))
                ])
                l2 = np.sqrt((candidates ** 2).sum(axis=1))
                fitness = -corr + penalty_val * n_viol + l2_lambda * l2
                es.tell(candidates, fitness)

                if variant == "sep":
                    es.C = np.diag(np.diag(es.C))
                    es._stale = True
                elif variant == "isotropic":
                    es.C = np.eye(es.d)
                    es._stale = True

                # track mutation acceptance
                current_best = es.best_ever[0] if es.best_ever else None
                if best_fitness_prev is not None and current_best is not None:
                    if current_best < best_fitness_prev - 1e-10:
                        mutations_accepted += 1
                best_fitness_prev = current_best
                total_gens += 1

                if es.should_restart():
                    es.restart()
                    if variant == "sep":
                        es.C = np.diag(np.diag(es.C))
                        es._stale = True
                    elif variant == "isotropic":
                        es.C = np.eye(es.d)
                        es._stale = True

                # trajectory at intervals
                if (gen + 1) % 500 == 0:
                    best = es.best_ever[1] if es.best_ever else es.mean
                    imp = feature_importance(best, D)
                    phi = float(imp[si])
                    regime = "A" if phi > 0.5 else ("B" if phi > 0.05 else "C")

                    mu_b_k = knuth_eval.evaluate_batch(best[None, :])
                    corr_b = spearman_batch_gpu(mu_b_k, target_ranks)

                    trajectory.append({
                        "gen": gen + 1,
                        "phi_sens": phi,
                        "regime": regime,
                        "r": float(corr_b[0]),
                        "sigma": float(es.sigma),
                        "mutations_accepted_cumul": mutations_accepted,
                    })

            elapsed = time.time() - t0
            final_genome = es.best_ever[1] if es.best_ever else es.mean
            imp_post = feature_importance(final_genome, D)
            phi_post = float(imp_post[si])

            result = {
                "seed": seed_id,
                "condition": cond_name,
                "variant": variant,
                "sigma0": sigma0,
                "phi_sens_pre": phi_pre,
                "phi_sens_post": phi_post,
                "regime_pre": "A" if phi_pre > 0.5 else ("B" if phi_pre > 0.05 else "C"),
                "regime_post": "A" if phi_post > 0.5 else ("B" if phi_post > 0.05 else "C"),
                "returned_to_c": phi_post <= 0.05,
                "mutations_accepted": mutations_accepted,
                "total_generations": total_gens,
                "mutation_acceptance_rate": mutations_accepted / max(total_gens, 1),
                "elapsed_s": elapsed,
                "trajectory": trajectory,
            }
            results[key] = result
            save_results(results, ESCAPE_PATH)

            log.info("  [done] %s — phi %.3f->%.3f, mutations_accepted=%d/%d (%.1fs)",
                     key, phi_pre, phi_post, mutations_accepted, total_gens, elapsed)

    # summary
    log.info("\nEscape attempt summary:")
    returned = sum(1 for v in results.values()
                   if isinstance(v, dict) and v.get("returned_to_c"))
    total = sum(1 for v in results.values()
                if isinstance(v, dict) and "returned_to_c" in v)
    avg_accept = np.mean([
        v["mutation_acceptance_rate"]
        for v in results.values()
        if isinstance(v, dict) and "mutation_acceptance_rate" in v
    ]) if total else 0
    log.info("  Returned to C: %d / %d", returned, total)
    log.info("  Mean mutation acceptance rate: %.4f", avg_accept)

    save_results(results, ESCAPE_PATH)


def show_status():
    for label, path in [("Lambda", LAMBDA_PATH), ("Escape", ESCAPE_PATH)]:
        results = load_results(path)
        count = sum(1 for v in results.values() if isinstance(v, dict))
        print(f"  {label}: {count} entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["lambda", "escape", "all", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    elif args.mode == "lambda":
        run_lambda()
    elif args.mode == "escape":
        run_escape()
    elif args.mode == "all":
        run_lambda()
        run_escape()


if __name__ == "__main__":
    main()
