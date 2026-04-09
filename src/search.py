"""
Step 2: CMA-ES evolutionary search for barrier-aware complexity measures.

Full checkpoint/resume support — can stop and restart at any generation.
GPU-accelerated fitness evaluation via batched matrix ops on RTX 3090.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import torch

from measures import (
    MeasureEvaluator, genome_dim, spearman_batch_gpu, precompute_target_ranks,
)
from barriers import check_all_barriers_batch, RelativizationChecker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "search.log")),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CMA-ES  (Hansen 2016, "The CMA Evolution Strategy: A Tutorial")
# ═══════════════════════════════════════════════════════════════════════════

class CMAES:
    """Self-contained CMA-ES with full state serialisation."""

    def __init__(self, d, *, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                 pop_size=None, seed=42):
        self.d = d
        self.sigma0 = sigma0
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.lam = pop_size or (4 + int(3 * np.log(d)))
        self.mu = self.lam // 2
        self.n_restarts = 0

        # recombination weights
        raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw / raw.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # learning rates
        self.c_sigma = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (d + 1)) - 1)
                        + self.c_sigma)
        self.c_c = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        self.c_1 = 2 / ((d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                        / ((d + 2) ** 2 + self.mu_eff))
        self.chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # mutable state
        self.mean = np.zeros(d)
        self.sigma = sigma0
        self.C = np.eye(d)
        self.p_sigma = np.zeros(d)
        self.p_c = np.zeros(d)
        self.gen = 0
        self.rng = np.random.RandomState(seed)
        self.best_ever = None    # (fitness, genome)

        # eigen cache
        self._evals = np.ones(d)
        self._evecs = np.eye(d)
        self._stale = True

    # -- sampling ----------------------------------------------------------

    def ask(self):
        if self._stale:
            self._evals, self._evecs = np.linalg.eigh(self.C)
            self._evals = np.maximum(self._evals, 1e-20)
            self._stale = False
        D_sqrt = np.sqrt(self._evals)
        z = self.rng.randn(self.lam, self.d)
        return self.mean + self.sigma * (z * D_sqrt) @ self._evecs.T

    # -- update (minimisation) ---------------------------------------------

    def tell(self, candidates, fitnesses):
        order = np.argsort(fitnesses)
        sel = candidates[order[: self.mu]]

        old_mean = self.mean.copy()
        self.mean = self.weights @ sel

        delta = (self.mean - old_mean) / self.sigma

        # C^{-1/2}
        inv_sqrt_D = 1.0 / np.sqrt(self._evals)
        C_invsqrt = (self._evecs * inv_sqrt_D) @ self._evecs.T

        # cumulation: sigma path
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                        * C_invsqrt @ delta)

        norm_ps = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2 / (self.d + 1)) * self.chi_n
        gen_factor = 1 - (1 - self.c_sigma) ** (2 * (self.gen + 1))
        h_sigma = int(norm_ps / np.sqrt(gen_factor) < threshold)

        # cumulation: C path
        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma
                    * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
                    * delta)

        # covariance update
        artmp = (sel - old_mean) / self.sigma          # (mu, d)
        rank_mu = (artmp.T * self.weights) @ artmp     # (d, d)
        c1_adj = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        self.C = ((1 - self.c_1 - self.c_mu + c1_adj * self.c_1) * self.C
                  + self.c_1 * np.outer(self.p_c, self.p_c)
                  + self.c_mu * rank_mu)

        # step-size adaptation
        self.sigma *= np.exp(
            self.c_sigma / self.d_sigma * (norm_ps / self.chi_n - 1))
        self.sigma = min(self.sigma, self.sigma_max)

        # condition number guard: reset C if it degenerates
        cond = self._evals.max() / max(self._evals.min(), 1e-20)
        if cond > 1e14:
            self.C = np.eye(self.d)

        self._stale = True
        self.gen += 1

        # track best-ever
        best_idx = order[0]
        if self.best_ever is None or fitnesses[best_idx] < self.best_ever[0]:
            self.best_ever = (float(fitnesses[best_idx]),
                              candidates[best_idx].copy())

    def should_restart(self):
        return self.sigma < self.sigma_min

    def restart(self):
        """Restart: keep best mean, reset sigma/C/paths. Explore new basin."""
        self.n_restarts += 1
        # alternate: even restarts exploit (keep mean), odd explore (random)
        if self.n_restarts % 2 == 0 and self.best_ever is not None:
            self.mean = self.best_ever[1].copy()
        else:
            self.mean = self.rng.randn(self.d) * self.sigma0
        self.sigma = self.sigma0
        self.C = np.eye(self.d)
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)
        self._evals = np.ones(self.d)
        self._evecs = np.eye(self.d)
        self._stale = True

    # -- checkpoint --------------------------------------------------------

    def save(self, path):
        state = {
            "mean": self.mean, "sigma": self.sigma, "C": self.C,
            "p_sigma": self.p_sigma, "p_c": self.p_c,
            "gen": self.gen, "n_restarts": self.n_restarts,
            "evals": self._evals, "evecs": self._evecs,
        }
        if self.best_ever is not None:
            state["best_ever_fit"] = self.best_ever[0]
            state["best_ever_genome"] = self.best_ever[1]
        np.savez(path, **state)
        rng_path = path.replace(".npz", "_rng.pkl")
        with open(rng_path, "wb") as f:
            pickle.dump(self.rng.get_state(), f)

    def load(self, path):
        s = np.load(path)
        self.mean = s["mean"]
        self.sigma = float(s["sigma"])
        self.C = s["C"]
        self.p_sigma = s["p_sigma"]
        self.p_c = s["p_c"]
        self.gen = int(s["gen"])
        self.n_restarts = int(s.get("n_restarts", 0))
        if "best_ever_fit" in s:
            self.best_ever = (float(s["best_ever_fit"]),
                              s["best_ever_genome"])
        self._evals = s["evals"]
        self._evecs = s["evecs"]
        self._stale = True
        rng_path = path.replace(".npz", "_rng.pkl")
        if os.path.exists(rng_path):
            with open(rng_path, "rb") as f:
                self.rng.set_state(pickle.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# search loop
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    path = os.path.join(PROJECT_DIR, "data", "search_features.npz")
    log.info("Loading %s", path)
    d = np.load(path)
    return d


def pick_device(cfg_device):
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def evaluate_population(candidates, knuth_eval, random_eval,
                        target_ranks, rel_checker, cfg, n_features):
    """Return (fitnesses_to_minimise, gen_stats)."""
    pop = len(candidates)

    # GPU batch evaluation
    mu_knuth = knuth_eval.evaluate_batch(candidates)    # (pop, N_k)
    mu_random = random_eval.evaluate_batch(candidates)  # (pop, N_r)

    # GPU Spearman
    corr = spearman_batch_gpu(mu_knuth, target_ranks)   # (pop,)

    # batch barrier checks
    barrier_pass, barrier_details = check_all_barriers_batch(
        mu_knuth, mu_random, candidates, rel_checker,
        cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=n_features,
    )
    n_violations = np.zeros(pop, dtype=np.int32)
    for i in range(pop):
        if not barrier_pass[i]:
            n_violations[i] = sum(
                1 for v in barrier_details[i].values() if not v["passes"])

    penalty = cfg["search"]["barrier_penalty"]
    l2_lambda = cfg["search"].get("l2_lambda", 0.0)
    l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
    fitness = -corr + penalty * n_violations + l2_lambda * l2_norm

    stats = {
        "best_corr": float(corr.max()),
        "mean_corr": float(corr.mean()),
        "best_fitness": float(fitness.min()),
        "barrier_pass_rate": float(barrier_pass.mean()),
        "n_feasible": int(barrier_pass.sum()),
        "sigma": None,  # filled by caller
    }
    return fitness, corr, barrier_pass, barrier_details, stats


def save_search_checkpoint(es, gen_log, top_candidates, cfg):
    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints")
    es.save(os.path.join(ckpt_dir, "cmaes_state.npz"))
    meta = {
        "generation": es.gen,
        "gen_log": gen_log,
        "top_candidates": top_candidates,
        "config_snapshot": cfg,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(ckpt_dir, "search_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_search_checkpoint(es):
    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints")
    cma_path = os.path.join(ckpt_dir, "cmaes_state.npz")
    meta_path = os.path.join(ckpt_dir, "search_meta.json")
    if not os.path.exists(cma_path) or not os.path.exists(meta_path):
        return None
    es.load(cma_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--max-gen", type=int, default=None,
                        help="Override max generations")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        cfg = json.load(f)

    if args.max_gen:
        cfg["search"]["max_generations"] = args.max_gen

    device = pick_device(cfg["search"]["device"])
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s, VRAM: %.1f GB",
                 torch.cuda.get_device_name(),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # load data
    data = load_data()
    knuth_m = data["knuth_measures"]
    knuth_t = data["knuth_targets"]
    knuth_q = data["knuth_quad"]
    knuth_bits = data["knuth_bits"]
    random_m = data["random_measures"]
    random_q = data["random_quad"]

    n_features = knuth_m.shape[1]
    g_dim = genome_dim(n_features)
    log.info("Functions: %d Knuth, %d random | Features: %d | Genome: %d",
             knuth_m.shape[0], random_m.shape[0], n_features, g_dim)

    # build evaluators
    knuth_eval = MeasureEvaluator(knuth_m, knuth_q, device)
    random_eval = MeasureEvaluator(random_m, random_q, device)
    target_ranks = precompute_target_ranks(knuth_t, device)

    # precompute relativization checker
    log.info("Precomputing relativization pseudoinverse…")
    rel_checker = RelativizationChecker(
        knuth_bits,
        poly_degree=cfg["barriers"]["relativization"]["poly_degree"],
        n_samples=cfg["barriers"]["relativization"]["n_samples"],
    )

    # CMA-ES
    pop_size = cfg["search"]["population_size"]
    es = CMAES(g_dim, sigma0=cfg["search"]["initial_sigma"],
               sigma_max=cfg["search"].get("sigma_max", 100.0),
               sigma_min=cfg["search"].get("sigma_min", 1e-6),
               pop_size=pop_size, seed=cfg["data"]["random_seed"])

    gen_log = []
    top_k = cfg["search"]["top_k"]
    top_candidates = []  # list of (fitness, corr, genome, barrier_details, gen)

    # resume?
    if args.resume:
        meta = load_search_checkpoint(es)
        if meta:
            gen_log = meta["gen_log"]
            top_candidates = meta["top_candidates"]
            log.info("Resumed from generation %d (%d log entries, %d top candidates)",
                     es.gen, len(gen_log), len(top_candidates))
        else:
            log.warning("No checkpoint found — starting fresh")

    max_gen = cfg["search"]["max_generations"]
    ckpt_every = cfg["search"]["checkpoint_every"]
    target_fit = cfg["search"]["target_fitness"]

    log.info("=== Search: pop=%d, gen=%d→%d, sigma=%.3f ===",
             pop_size, es.gen, max_gen, es.sigma)

    t_start = time.time()

    while es.gen < max_gen:
        t_gen = time.time()
        candidates = es.ask()

        fitness, corr, bp, bd, stats = evaluate_population(
            candidates, knuth_eval, random_eval,
            target_ranks, rel_checker, cfg, n_features,
        )
        es.tell(candidates, fitness)

        # sigma restart
        if es.should_restart():
            es.restart()
            log.info("*** RESTART #%d at gen %d (sigma collapsed) ***",
                     es.n_restarts, es.gen)

        stats["sigma"] = float(es.sigma)
        stats["generation"] = es.gen
        stats["n_restarts"] = es.n_restarts
        stats["elapsed_s"] = time.time() - t_gen
        gen_log.append(stats)

        # update top-k
        for i in range(len(candidates)):
            entry = {
                "fitness": float(fitness[i]),
                "corr": float(corr[i]),
                "genome": candidates[i].tolist(),
                "barrier_details": bd[i],
                "generation": es.gen,
                "feasible": bool(bp[i]),
            }
            top_candidates.append(entry)
        top_candidates.sort(key=lambda x: x["fitness"])
        top_candidates = top_candidates[:top_k]

        # logging
        if es.gen % 10 == 0 or es.gen <= 5:
            log.info(
                "gen %5d | best_r %.4f  mean_r %.4f | feasible %d/%d | "
                "sigma %.3e | %.1fs",
                es.gen, stats["best_corr"], stats["mean_corr"],
                stats["n_feasible"], pop_size, es.sigma,
                stats["elapsed_s"],
            )

        # checkpoint
        if es.gen % ckpt_every == 0:
            save_search_checkpoint(es, gen_log, top_candidates, cfg)

        # early stop
        if stats["best_corr"] >= target_fit and stats["n_feasible"] > 0:
            log.info("Target fitness %.2f reached at gen %d — stopping",
                     target_fit, es.gen)
            break

    # final save
    save_search_checkpoint(es, gen_log, top_candidates, cfg)

    total = time.time() - t_start
    log.info("=== Search complete: %d generations in %.1f min ===",
             es.gen, total / 60)

    # save top candidates separately
    out_path = os.path.join(PROJECT_DIR, "data", "top_candidates.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_generations": es.gen,
            "total_time_s": total,
            "top_k": top_k,
            "candidates": top_candidates,
        }, f, indent=2)
    log.info("Top %d candidates saved to %s", len(top_candidates), out_path)


if __name__ == "__main__":
    main()
