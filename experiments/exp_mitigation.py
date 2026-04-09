"""
Block 7C: Adaptive sep-switching mitigation (proof of concept).

Strategy: monitor phi_sens every 100 generations. If phi_sens increases by
> 0.03 over a 500-generation window, switch from full to sep-CMA-ES.

This gives the paper a "problem + preliminary solution" structure.

Usage:
    python exp_mitigation.py run
    python exp_mitigation.py status
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

log = get_logger("exp_mitigation")

RESULT_PATH = os.path.join(RESULTS_DIR, "adaptive_mitigation.json")

SEEDS = list(range(1, 31))
MAX_GEN = 3000
POP_SIZE = 50
CHECKPOINTS = [500, 1000, 2000, 3000]

# mitigation parameters
MONITOR_INTERVAL = 100      # check phi_sens every N gen
WINDOW = 500                # look-back window for drift detection
DRIFT_THRESHOLD = 0.03      # phi_sens increase that triggers switch
SWITCH_BACK_AFTER = 1000    # try switching back to full after N gen


def run_adaptive_trial(seed, domain):
    """Run CMA-ES with adaptive full->sep switching on drift detection."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
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

    es = CMAES(GENOME_DIM, sigma0=0.3, sigma_max=100.0, sigma_min=1e-6,
               pop_size=POP_SIZE, seed=seed)

    penalty = cfg["search"]["barrier_penalty"]
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    si = 8

    current_mode = "full"
    switch_events = []
    phi_history = []
    gen_switched_to_sep = None
    checkpoint_set = set(CHECKPOINTS)
    trajectory = []
    t0 = time.time()

    for gen in range(MAX_GEN):
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
        fitness = -corr + penalty * n_viol + l2_lambda * l2
        es.tell(candidates, fitness)

        # enforce current mode
        if current_mode == "sep":
            es.C = np.diag(np.diag(es.C))
            es._stale = True

        if es.should_restart():
            es.restart()
            if current_mode == "sep":
                es.C = np.diag(np.diag(es.C))
                es._stale = True

        # adaptive monitoring
        if (gen + 1) % MONITOR_INTERVAL == 0:
            best = es.best_ever[1] if es.best_ever else es.mean
            imp = feature_importance(best, D)
            phi = float(imp[si])
            phi_history.append((gen + 1, phi))

            # check for drift
            if current_mode == "full" and len(phi_history) >= WINDOW // MONITOR_INTERVAL + 1:
                lookback_gen = gen + 1 - WINDOW
                past_phi = [p for g, p in phi_history if g <= lookback_gen]
                if past_phi:
                    delta = phi - past_phi[-1]
                    if delta > DRIFT_THRESHOLD:
                        log.info("    [switch] gen %d: phi_sens %.3f -> %.3f "
                                 "(delta=%.3f > %.3f), switching to sep",
                                 gen + 1, past_phi[-1], phi, delta, DRIFT_THRESHOLD)
                        current_mode = "sep"
                        gen_switched_to_sep = gen + 1
                        switch_events.append({
                            "gen": gen + 1, "direction": "full->sep",
                            "phi_sens": phi, "delta": delta,
                        })
                        es.C = np.diag(np.diag(es.C))
                        es._stale = True

            # try switching back
            if (current_mode == "sep" and gen_switched_to_sep is not None
                    and gen + 1 - gen_switched_to_sep >= SWITCH_BACK_AFTER):
                if phi <= 0.03:
                    log.info("    [switch] gen %d: phi_sens=%.3f low, switching back to full",
                             gen + 1, phi)
                    current_mode = "full"
                    gen_switched_to_sep = None
                    switch_events.append({
                        "gen": gen + 1, "direction": "sep->full",
                        "phi_sens": phi,
                    })

        # trajectory checkpoint
        if (gen + 1) in checkpoint_set:
            best = es.best_ever[1] if es.best_ever else es.mean
            imp = feature_importance(best, D)
            phi = float(imp[si])
            regime = "A" if phi > 0.5 else ("B" if phi > 0.05 else "C")

            mu_b_k = knuth_eval.evaluate_batch(best[None, :])
            mu_b_r = random_eval.evaluate_batch(best[None, :])
            corr_b = spearman_batch_gpu(mu_b_k, target_ranks)
            bp_b, bd_b = check_all_barriers_batch(
                mu_b_k, mu_b_r, best[None, :], rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )

            trajectory.append({
                "gen": gen + 1,
                "r": float(corr_b[0]),
                "phi_sens": phi,
                "regime": regime,
                "feasible": bool(bp_b[0]),
                "np_density": float(bd_b[0]["natural_proof"]["density"]),
                "mode": current_mode,
                "n_switches": len(switch_events),
            })

    elapsed = time.time() - t0
    final = trajectory[-1] if trajectory else {}

    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "final": final,
        "trajectory": trajectory,
        "switch_events": switch_events,
        "phi_history": phi_history,
    }


def run():
    log.info("=" * 70)
    log.info("Block 7C: Adaptive sep-switching mitigation")
    log.info("=" * 70)

    results = load_results(RESULT_PATH)
    domain = setup_boolean_domain()

    for seed in SEEDS:
        key = f"adaptive_seed{seed}"
        if key in results:
            log.info("  [skip] %s", key)
            continue

        log.info("  [run]  %s ...", key)
        trial = run_adaptive_trial(seed, domain)
        results[key] = trial
        save_results(results, RESULT_PATH)
        log.info("  [done] %s — regime=%s, r=%.4f, switches=%d (%.1fs)",
                 key, trial["final"].get("regime", "?"),
                 trial["final"].get("r", 0),
                 len(trial["switch_events"]),
                 trial["elapsed_s"])

    # summary: compare adaptive vs pure full (from 30-seed experiment)
    log.info("\nAdaptive mitigation summary:")
    adaptive_regimes = []
    for seed in SEEDS:
        key = f"adaptive_seed{seed}"
        if key in results:
            adaptive_regimes.append(results[key]["final"].get("regime", "?"))

    if adaptive_regimes:
        n = len(adaptive_regimes)
        log.info("  Adaptive: A=%d B=%d C=%d (n=%d)",
                 adaptive_regimes.count("A"),
                 adaptive_regimes.count("B"),
                 adaptive_regimes.count("C"), n)
        drift = sum(1 for r in adaptive_regimes if r in ("A", "B"))
        log.info("  Drift rate: %d/%d (%.0f%%)", drift, n, 100 * drift / n)

        # mean rho
        rhos = [results[f"adaptive_seed{s}"]["final"]["r"]
                for s in SEEDS if f"adaptive_seed{s}" in results
                and "r" in results[f"adaptive_seed{s}"]["final"]]
        if rhos:
            log.info("  Mean rho: %.4f", np.mean(rhos))

    save_results(results, RESULT_PATH)


def show_status():
    results = load_results(RESULT_PATH)
    done = sum(1 for s in SEEDS if f"adaptive_seed{s}" in results)
    print(f"  Adaptive mitigation: {done}/{len(SEEDS)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    else:
        run()


if __name__ == "__main__":
    main()
