"""
Shared utilities for experiments.
Checkpoint-resumable, JSON-serialisable results.
"""

import json
import logging
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "jmlr_revision")

sys.path.insert(0, SRC_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PROJECT_DIR, "logs", f"{name}.log")),
        ],
    )
    return logging.getLogger(name)


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def result_key(variant, seed):
    return f"{variant}_seed{seed}"


def setup_boolean_domain():
    """Load the Boolean function complexity domain data + evaluators."""
    import torch
    from mechanism import MEASURE_NAMES, D, Q_DIM, GENOME_DIM, load_data
    from measures import MeasureEvaluator, precompute_target_ranks
    from barriers import RelativizationChecker

    measures, quad, targets = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    knuth_eval = MeasureEvaluator(measures, quad, device)
    target_ranks = precompute_target_ranks(targets, device)

    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        cfg = json.load(f)

    data = np.load(os.path.join(DATA_DIR, "search_features.npz"))
    random_eval = MeasureEvaluator(
        data["random_measures"], data["random_quad"], device)
    rel_checker = RelativizationChecker(
        data["knuth_bits"],
        poly_degree=cfg["barriers"]["relativization"]["poly_degree"],
        n_samples=cfg["barriers"]["relativization"]["n_samples"],
    )

    return {
        "measures": measures, "quad": quad, "targets": targets,
        "knuth_eval": knuth_eval, "random_eval": random_eval,
        "target_ranks": target_ranks, "rel_checker": rel_checker,
        "cfg": cfg, "data": data, "device": device,
        "D": D, "Q_DIM": Q_DIM, "GENOME_DIM": GENOME_DIM,
        "MEASURE_NAMES": MEASURE_NAMES,
    }


def classify_genome(genome, knuth_eval, random_eval,
                    target_ranks, rel_checker, cfg, D=10):
    """Classify a genome into Type A/B/C with full diagnostics."""
    import torch
    from measures import spearman_batch_gpu
    from barriers import check_all_barriers_batch
    from analyze_candidates import feature_importance

    si = cfg["_measure_names"].index("sensitivity") if "_measure_names" in cfg \
        else 8  # storage index in code order

    mu_k = knuth_eval.evaluate_batch(genome[None, :])
    mu_r = random_eval.evaluate_batch(genome[None, :])
    corr = spearman_batch_gpu(mu_k, target_ranks)
    bp, bd = check_all_barriers_batch(
        mu_k, mu_r, genome[None, :], rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
    )

    imp = feature_importance(genome, D)
    phi_sens = float(imp[si])
    np_density = float(bd[0]["natural_proof"]["density"])

    if phi_sens > 0.05:
        if phi_sens > 0.5:
            regime = "A"
        else:
            regime = "B"
    else:
        regime = "C"

    return {
        "regime": regime,
        "r": float(corr[0]),
        "phi_sens": phi_sens,
        "np_density": np_density,
        "feasible": bool(bp[0]),
        "barrier_status": bd[0],
    }


def run_cmaes_trial(seed, variant, max_gen, pop_size, domain,
                    checkpoints=None, sigma0=0.3):
    """Run a single CMA-ES trial with variant enforcement.

    variant: "full", "sep", or "isotropic"
    checkpoints: list of generation numbers to record diagnostics
    Returns: dict with final state + trajectory
    """
    from search import CMAES
    from measures import spearman_batch_gpu
    from barriers import check_all_barriers_batch
    from analyze_candidates import feature_importance

    if checkpoints is None:
        checkpoints = [max_gen]
    checkpoint_set = set(checkpoints)

    D = domain["D"]
    GENOME_DIM = domain["GENOME_DIM"]
    knuth_eval = domain["knuth_eval"]
    random_eval = domain["random_eval"]
    target_ranks = domain["target_ranks"]
    rel_checker = domain["rel_checker"]
    cfg = domain["cfg"]

    es = CMAES(GENOME_DIM, sigma0=sigma0, sigma_max=100.0, sigma_min=1e-6,
               pop_size=pop_size, seed=seed)

    penalty = cfg["search"]["barrier_penalty"]
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    si = 8  # sensitivity storage index

    trajectory = []
    t0 = time.time()

    for gen in range(max_gen):
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

        # enforce variant
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

        # checkpoint diagnostics
        if (gen + 1) in checkpoint_set or gen == max_gen - 1:
            best_genome = es.best_ever[1] if es.best_ever else es.mean
            imp = feature_importance(best_genome, D)
            phi_sens = float(imp[si])

            mu_best_k = knuth_eval.evaluate_batch(best_genome[None, :])
            mu_best_r = random_eval.evaluate_batch(best_genome[None, :])
            corr_best = spearman_batch_gpu(mu_best_k, target_ranks)
            bp_best, bd_best = check_all_barriers_batch(
                mu_best_k, mu_best_r, best_genome[None, :], rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )

            regime = "A" if phi_sens > 0.5 else ("B" if phi_sens > 0.05 else "C")
            trajectory.append({
                "gen": gen + 1,
                "r": float(corr_best[0]),
                "phi_sens": phi_sens,
                "regime": regime,
                "feasible": bool(bp_best[0]),
                "np_density": float(bd_best[0]["natural_proof"]["density"]),
                "sigma": float(es.sigma),
                "n_restarts": es.n_restarts,
            })

    elapsed = time.time() - t0
    final = trajectory[-1] if trajectory else {}

    return {
        "seed": seed,
        "variant": variant,
        "max_gen": max_gen,
        "pop_size": pop_size,
        "elapsed_s": elapsed,
        "final": final,
        "trajectory": trajectory,
        "genome": es.best_ever[1].tolist() if es.best_ever else None,
    }
