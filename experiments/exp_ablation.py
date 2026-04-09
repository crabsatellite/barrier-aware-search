"""
Block 3A + 3B: Same-problem ablations for C1 and C3.

3A: Replace Spearman (rank-based, non-smooth) with MSE (smooth) — tests C1.
3B: Replace binary constraints with soft penalty — tests C3.

30 seeds x 3 variants x 8000 gen on the original Boolean domain.

Usage:
    python exp_ablation.py c1          # Spearman -> MSE (Block 3A)
    python exp_ablation.py c3          # binary -> soft penalty (Block 3B)
    python exp_ablation.py all
    python exp_ablation.py status
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

log = get_logger("exp_ablation")

C1_PATH = os.path.join(RESULTS_DIR, "ablation_c1_mse.json")
C3_PATH = os.path.join(RESULTS_DIR, "ablation_c3_soft.json")

SEEDS = list(range(1, 31))
VARIANTS = ["full", "sep", "isotropic"]
MAX_GEN = 3000
POP_SIZE = 50
CHECKPOINTS = [500, 1000, 2000, 3000]


def run_c1_trial(seed, variant, domain):
    """C1 ablation: replace Spearman with MSE objective."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from search import CMAES
    from barriers import check_all_barriers_batch
    from analyze_candidates import feature_importance

    cfg = domain["cfg"]
    knuth_eval = domain["knuth_eval"]
    random_eval = domain["random_eval"]
    rel_checker = domain["rel_checker"]
    D = domain["D"]
    GENOME_DIM = domain["GENOME_DIM"]
    targets = domain["targets"]

    # normalise targets for MSE
    t_mean = targets.mean()
    t_std = targets.std() + 1e-8
    targets_norm = (targets - t_mean) / t_std

    es = CMAES(GENOME_DIM, sigma0=0.3, sigma_max=100.0, sigma_min=1e-6,
               pop_size=POP_SIZE, seed=seed)

    penalty = cfg["search"]["barrier_penalty"]
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    si = 8
    checkpoint_set = set(CHECKPOINTS)
    trajectory = []
    t0 = time.time()

    for gen in range(MAX_GEN):
        candidates = es.ask()

        mu_k = knuth_eval.evaluate_batch(candidates)
        mu_r = random_eval.evaluate_batch(candidates)

        # MSE instead of Spearman
        mu_std = np.std(mu_k, axis=1, keepdims=True) + 1e-8
        mu_mean = np.mean(mu_k, axis=1, keepdims=True)
        mu_norm = (mu_k - mu_mean) / mu_std
        mse = np.mean((mu_norm - targets_norm[None, :]) ** 2, axis=1)
        quality = -mse  # higher is better (negated for fitness minimisation)

        bp, bd = check_all_barriers_batch(
            mu_k, mu_r, candidates, rel_checker,
            cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
        )
        n_viol = np.array([
            sum(1 for v in bd[i].values() if not v["passes"])
            for i in range(len(candidates))
        ])

        l2 = np.sqrt((candidates ** 2).sum(axis=1))
        fitness = mse + penalty * n_viol + l2_lambda * l2
        es.tell(candidates, fitness)

        if variant == "sep":
            es.C = np.diag(np.diag(es.C))
            es._stale = True
        elif variant == "isotropic":
            es.C = np.eye(es.d)
            es._stale = True

        if es.should_restart():
            es.restart()
            if variant == "sep":
                es.C = np.diag(np.diag(es.C))
                es._stale = True
            elif variant == "isotropic":
                es.C = np.eye(es.d)
                es._stale = True

        if (gen + 1) in checkpoint_set:
            best = es.best_ever[1] if es.best_ever else es.mean
            imp = feature_importance(best, D)
            phi_sens = float(imp[si])
            regime = "A" if phi_sens > 0.5 else ("B" if phi_sens > 0.05 else "C")

            mu_best = knuth_eval.evaluate_batch(best[None, :])
            mu_best_r = random_eval.evaluate_batch(best[None, :])
            bp_b, bd_b = check_all_barriers_batch(
                mu_best, mu_best_r, best[None, :], rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )

            trajectory.append({
                "gen": gen + 1,
                "mse": float(np.mean((
                    (mu_best[0] - mu_best[0].mean()) / (mu_best[0].std() + 1e-8)
                    - targets_norm) ** 2)),
                "phi_sens": phi_sens,
                "regime": regime,
                "feasible": bool(bp_b[0]),
                "np_density": float(bd_b[0]["natural_proof"]["density"]),
            })

    return {
        "seed": seed, "variant": variant,
        "elapsed_s": time.time() - t0,
        "final": trajectory[-1] if trajectory else {},
        "trajectory": trajectory,
    }


def run_c3_trial(seed, variant, domain):
    """C3 ablation: replace binary constraints with soft penalty."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from search import CMAES
    from measures import spearman_batch_gpu
    from analyze_candidates import feature_importance

    cfg = domain["cfg"]
    knuth_eval = domain["knuth_eval"]
    random_eval = domain["random_eval"]
    target_ranks = domain["target_ranks"]
    rel_checker = domain["rel_checker"]
    D = domain["D"]
    GENOME_DIM = domain["GENOME_DIM"]

    es = CMAES(GENOME_DIM, sigma0=0.3, sigma_max=100.0, sigma_min=1e-6,
               pop_size=POP_SIZE, seed=seed)

    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    n_vars = cfg["data"]["n_vars"]
    tau = 1.0 / (n_vars ** 2)  # 0.04
    soft_penalty_coeff = 500.0  # strong but continuous
    si = 8
    checkpoint_set = set(CHECKPOINTS)
    trajectory = []
    t0 = time.time()

    for gen in range(MAX_GEN):
        candidates = es.ask()

        mu_k = knuth_eval.evaluate_batch(candidates)
        mu_r = random_eval.evaluate_batch(candidates)
        corr = spearman_batch_gpu(mu_k, target_ranks)

        # soft NP penalty instead of binary
        medians = np.median(np.abs(mu_k), axis=1)
        above = np.abs(mu_r) >= medians[:, None]
        densities = above.astype(np.float64).mean(axis=1)
        excess = np.maximum(densities - tau, 0)
        soft_penalty = soft_penalty_coeff * excess ** 2

        l2 = np.sqrt((candidates ** 2).sum(axis=1))
        fitness = -corr + soft_penalty + l2_lambda * l2
        es.tell(candidates, fitness)

        if variant == "sep":
            es.C = np.diag(np.diag(es.C))
            es._stale = True
        elif variant == "isotropic":
            es.C = np.eye(es.d)
            es._stale = True

        if es.should_restart():
            es.restart()
            if variant == "sep":
                es.C = np.diag(np.diag(es.C))
                es._stale = True
            elif variant == "isotropic":
                es.C = np.eye(es.d)
                es._stale = True

        if (gen + 1) in checkpoint_set:
            best = es.best_ever[1] if es.best_ever else es.mean
            imp = feature_importance(best, D)
            phi_sens = float(imp[si])
            regime = "A" if phi_sens > 0.5 else ("B" if phi_sens > 0.05 else "C")

            mu_best = knuth_eval.evaluate_batch(best[None, :])
            mu_best_r = random_eval.evaluate_batch(best[None, :])
            corr_best = spearman_batch_gpu(mu_best, target_ranks)
            med_best = np.median(np.abs(mu_best[0]))
            dens_best = float((np.abs(mu_best_r[0]) >= med_best).mean())

            trajectory.append({
                "gen": gen + 1,
                "r": float(corr_best[0]),
                "phi_sens": phi_sens,
                "regime": regime,
                "np_density": dens_best,
                "feasible": dens_best <= tau,
            })

    return {
        "seed": seed, "variant": variant,
        "elapsed_s": time.time() - t0,
        "final": trajectory[-1] if trajectory else {},
        "trajectory": trajectory,
    }


def run_experiment(ablation_type):
    if ablation_type == "c1":
        path = C1_PATH
        trial_fn = run_c1_trial
        label = "C1 (Spearman -> MSE)"
    else:
        path = C3_PATH
        trial_fn = run_c3_trial
        label = "C3 (binary -> soft penalty)"

    log.info("=" * 70)
    log.info("Block 3%s: %s ablation", "A" if ablation_type == "c1" else "B", label)
    log.info("=" * 70)

    results = load_results(path)
    domain = setup_boolean_domain()

    for seed in SEEDS:
        for variant in VARIANTS:
            key = result_key(variant, seed)
            if key in results:
                log.info("  [skip] %s", key)
                continue

            log.info("  [run]  %s ...", key)
            trial = trial_fn(seed, variant, domain)
            results[key] = trial
            save_results(results, path)
            log.info("  [done] %s — regime=%s, phi_sens=%.4f (%.1fs)",
                     key, trial["final"].get("regime", "?"),
                     trial["final"].get("phi_sens", 0), trial["elapsed_s"])

    _summarize(results, path, label)


def _summarize(results, path, label):
    from scipy.stats import fisher_exact
    log.info("\n%s summary:", label)
    for variant in VARIANTS:
        drift = sum(1 for s in SEEDS
                    if result_key(variant, s) in results
                    and results[result_key(variant, s)].get("final", {}).get("regime") in ("A", "B"))
        total = sum(1 for s in SEEDS if result_key(variant, s) in results)
        if total:
            log.info("  %-12s: drift %d/%d", variant, drift, total)

    # Fisher's exact
    fd = sum(1 for s in SEEDS
             if result_key("full", s) in results
             and results[result_key("full", s)].get("final", {}).get("regime") in ("A", "B"))
    ft = sum(1 for s in SEEDS if result_key("full", s) in results)
    sd = sum(1 for s in SEEDS
             if result_key("sep", s) in results
             and results[result_key("sep", s)].get("final", {}).get("regime") in ("A", "B"))
    st = sum(1 for s in SEEDS if result_key("sep", s) in results)
    if ft > 0 and st > 0:
        _, p = fisher_exact([[fd, ft - fd], [sd, st - sd]], alternative="greater")
        log.info("  Fisher (full vs sep, one-sided): p = %.4f", p)


def show_status():
    for label, path in [("C1 (MSE)", C1_PATH), ("C3 (soft)", C3_PATH)]:
        results = load_results(path)
        total = len(SEEDS) * len(VARIANTS)
        done = sum(1 for s in SEEDS for v in VARIANTS
                   if result_key(v, s) in results)
        print(f"  {label}: {done}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["c1", "c3", "all", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    elif args.mode == "c1":
        run_experiment("c1")
    elif args.mode == "c3":
        run_experiment("c3")
    elif args.mode == "all":
        run_experiment("c1")
        run_experiment("c3")


if __name__ == "__main__":
    main()
