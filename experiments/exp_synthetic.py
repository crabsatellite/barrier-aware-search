"""
Block 2C: Synthetic Rank-Barrier benchmark.

3x2x2 ablation matrix testing conditions C1, C2, C3 independently.

Axes:
  - Objective:   rank (Spearman) vs smooth (MSE)        → C1
  - Cross-terms: present vs absent                      → C2
  - Constraint:  binary (pass/fail) vs soft (penalty)   → C3

30 seeds per cell, 3 variants (full/sep/isotropic) = 12 cells x 30 x 3 = 1080 runs.
Each run is fast (~2-5 min on CPU).

Usage:
    python exp_synthetic.py run       # run all cells
    python exp_synthetic.py status    # show progress
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import fisher_exact

from shared import get_logger, RESULTS_DIR, load_results, save_results

log = get_logger("exp_synthetic")

RESULT_PATH = os.path.join(RESULTS_DIR, "synthetic_ablation.json")

N_SEEDS = 30
SEEDS = list(range(1, N_SEEDS + 1))
VARIANTS = ["full", "sep", "isotropic"]
MAX_GEN = 3000
POP_SIZE = 50


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic problem: Rank-Barrier benchmark
# ═══════════════════════════════════════════════════════════════════════════

def _fast_spearman_batch(X, target_ranks):
    """Vectorized Spearman correlation: each column of X vs target_ranks."""
    n = X.shape[0]
    x_ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    d = target_ranks[:, None] - x_ranks
    return 1.0 - 6.0 * (d ** 2).sum(axis=0) / (n * (n ** 2 - 1))


class RankBarrierProblem:
    """Parameterised synthetic problem with controllable C1/C2/C3."""

    def __init__(self, d=10, d_overhead=3, N=2000,
                 objective="rank", constraint="binary",
                 cross_terms=True, tau=0.05, seed=0):
        self.d = d
        self.d_signal = d - d_overhead
        self.d_overhead = d_overhead
        self.N = N
        self.objective = objective
        self.constraint = constraint
        self.cross_terms = cross_terms
        self.tau = tau

        rng = np.random.RandomState(seed + 10000)

        # generate synthetic items with feature correlations
        self.target_ranking = rng.permutation(N).astype(np.float64)
        self._target_ranks = np.argsort(np.argsort(self.target_ranking)).astype(np.float64)

        base = rng.randn(N, d).astype(np.float64)

        # inject target correlation into signal features
        for i in range(self.d_signal):
            strength = 0.3 + 0.4 * rng.rand()
            noise = rng.randn(N) * (1 - strength)
            base[:, i] = strength * self.target_ranking / N + noise

        # inject signal-overhead correlation (creates exploitable cross-dims)
        for i in range(self.d_signal, d):
            source = rng.randint(0, self.d_signal)
            corr_strength = 0.2 + 0.3 * rng.rand()
            base[:, i] = corr_strength * base[:, source] + (1 - corr_strength) * base[:, i]

        self.items = base

        # quadratic features (if cross_terms enabled)
        if cross_terms:
            pairs = []
            for i in range(d):
                for j in range(i, d):
                    pairs.append(base[:, i] * base[:, j])
            self.quad = np.column_stack(pairs)
            self.q_dim = len(pairs)
        else:
            diag_pairs = [base[:, i] ** 2 for i in range(d)]
            self.quad = np.column_stack(diag_pairs)
            self.q_dim = d

        # genome: linear(d) + quad(q_dim) + activation(d) + threshold(d)
        self.genome_dim = d + self.q_dim + d + d

        # random "population" for density constraint (analogous to NP barrier)
        n_rand = 1000
        self.random_items = rng.randn(n_rand, d).astype(np.float64)
        if cross_terms:
            pairs_r = []
            for i in range(d):
                for j in range(i, d):
                    pairs_r.append(self.random_items[:, i] * self.random_items[:, j])
            self.random_quad = np.column_stack(pairs_r)
        else:
            self.random_quad = np.column_stack(
                [self.random_items[:, i] ** 2 for i in range(d)])

    def evaluate_batch(self, genomes):
        """Evaluate a batch of genomes. Returns (fitness_to_minimise, diagnostics)."""
        d = self.d
        q = self.q_dim
        pop = len(genomes)

        W = genomes[:, :d]
        Q = genomes[:, d:d + q]
        A = genomes[:, d + q:d + q + d]
        T = genomes[:, d + q + d:d + q + 2 * d]

        # mu = W'x + Q'quad + A'relu(x - T)
        linear = self.items @ W.T  # (N, pop)
        quadratic = self.quad @ Q.T  # (N, pop)

        # threshold term — vectorized via broadcasting
        # items: (N,d), T: (pop,d) → diff: (N,pop,d)
        diff = self.items[:, None, :] - T[None, :, :]
        threshold = np.einsum('npd,pd->np', np.maximum(diff, 0), A)

        mu = linear + quadratic + threshold  # (N, pop)

        # objective
        if self.objective == "rank":
            corr = _fast_spearman_batch(mu, self._target_ranks)
        else:  # MSE
            target_norm = self.target_ranking / self.N
            mu_std = mu / (mu.std(axis=0, keepdims=True) + 1e-8)
            corr = -((mu_std - target_norm[:, None]) ** 2).mean(axis=0)

        # constraint: density of random items exceeding dataset median
        median_abs = np.median(np.abs(mu), axis=0)  # (pop,)

        # random item evaluation — vectorized
        lin_r = self.random_items @ W.T
        quad_r = self.random_quad @ Q.T
        diff_r = self.random_items[:, None, :] - T[None, :, :]
        thresh_r = np.einsum('npd,pd->np', np.maximum(diff_r, 0), A)
        mu_r = lin_r + quad_r + thresh_r
        density = (np.abs(mu_r) >= median_abs[None, :]).mean(axis=0)

        if self.constraint == "binary":
            violated = density > self.tau
            penalty = violated.astype(np.float64) * 10.0
        else:  # soft
            excess = np.maximum(density - self.tau, 0)
            penalty = 100.0 * excess ** 2

        l2 = np.sqrt((genomes ** 2).sum(axis=1))
        fitness = -corr + penalty + 0.001 * l2

        # overhead importance (analogous to phi_sens)
        overhead_mass = np.abs(genomes[:, self.d_signal:d]).sum(axis=1)
        total_mass = np.abs(genomes[:, :d]).sum(axis=1) + 1e-12
        phi_overhead = overhead_mass / total_mass

        return fitness, {
            "corr": corr,
            "density": density,
            "violated": violated if self.constraint == "binary" else (density > self.tau),
            "phi_overhead": phi_overhead,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CMA-ES runner (self-contained, no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleCMAES:
    """Minimal CMA-ES for synthetic experiments."""

    def __init__(self, d, sigma0=0.5, pop_size=None, seed=42):
        self.d = d
        self.lam = pop_size or (4 + int(3 * np.log(d)))
        self.mu = self.lam // 2
        raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw / raw.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        self.c_sigma = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        self.c_1 = 2 / ((d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                        / ((d + 2) ** 2 + self.mu_eff))
        self.chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        self.mean = np.zeros(d)
        self.sigma = sigma0
        self.C = np.eye(d)
        self.p_sigma = np.zeros(d)
        self.p_c = np.zeros(d)
        self.gen = 0
        self.rng = np.random.RandomState(seed)
        self.best_ever = None
        self._evals = np.ones(d)
        self._evecs = np.eye(d)
        self._stale = True

    def ask(self):
        if self._stale:
            self._evals, self._evecs = np.linalg.eigh(self.C)
            self._evals = np.maximum(self._evals, 1e-20)
            self._stale = False
        D_sqrt = np.sqrt(self._evals)
        z = self.rng.randn(self.lam, self.d)
        return self.mean + self.sigma * (z * D_sqrt) @ self._evecs.T

    def tell(self, candidates, fitnesses):
        order = np.argsort(fitnesses)
        sel = candidates[order[:self.mu]]
        old_mean = self.mean.copy()
        self.mean = self.weights @ sel
        delta = (self.mean - old_mean) / self.sigma

        inv_sqrt_D = 1.0 / np.sqrt(self._evals)
        C_invsqrt = (self._evecs * inv_sqrt_D) @ self._evecs.T

        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                        * C_invsqrt @ delta)
        norm_ps = np.linalg.norm(self.p_sigma)
        gen_factor = 1 - (1 - self.c_sigma) ** (2 * (self.gen + 1))
        h_sigma = int(norm_ps / np.sqrt(gen_factor) < (1.4 + 2 / (self.d + 1)) * self.chi_n)

        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * delta)

        artmp = (sel - old_mean) / self.sigma
        rank_mu = (artmp.T * self.weights) @ artmp
        c1_adj = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        self.C = ((1 - self.c_1 - self.c_mu + c1_adj * self.c_1) * self.C
                  + self.c_1 * np.outer(self.p_c, self.p_c)
                  + self.c_mu * rank_mu)

        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (norm_ps / self.chi_n - 1))
        self.sigma = min(self.sigma, 100.0)

        cond = self._evals.max() / max(self._evals.min(), 1e-20)
        if cond > 1e14:
            self.C = np.eye(self.d)

        self._stale = True
        self.gen += 1

        best_idx = order[0]
        if self.best_ever is None or fitnesses[best_idx] < self.best_ever[0]:
            self.best_ever = (float(fitnesses[best_idx]), candidates[best_idx].copy())

    def should_restart(self):
        return self.sigma < 1e-6

    def restart(self):
        self.sigma = 0.5
        self.C = np.eye(self.d)
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)
        self._stale = True
        if self.best_ever is not None:
            self.mean = self.best_ever[1].copy()


def run_synthetic_trial(problem, variant, seed, max_gen, pop_size):
    """Run one trial on a synthetic problem."""
    es = SimpleCMAES(problem.genome_dim, sigma0=0.5, pop_size=pop_size, seed=seed)
    d = problem.d
    checkpoints = [500, 1000, 2000, 3000]
    checkpoint_set = set(checkpoints)
    trajectory = []
    t0 = time.time()

    for gen in range(max_gen):
        candidates = es.ask()
        fitness, diag = problem.evaluate_batch(candidates)
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
            _, best_diag = problem.evaluate_batch(best[None, :])
            trajectory.append({
                "gen": gen + 1,
                "corr": float(best_diag["corr"][0]),
                "phi_overhead": float(best_diag["phi_overhead"][0]),
                "density": float(best_diag["density"][0]),
                "violated": bool(best_diag["violated"][0]),
                "regime": "B" if best_diag["phi_overhead"][0] > 0.05 else "C",
            })

    elapsed = time.time() - t0
    final = trajectory[-1] if trajectory else {}
    return {
        "seed": seed, "variant": variant,
        "elapsed_s": elapsed,
        "final": final,
        "trajectory": trajectory,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ablation matrix
# ═══════════════════════════════════════════════════════════════════════════

CELLS = [
    # (label, objective, cross_terms, constraint, d, d_oh) — C1/C2/C3 annotation
    ("rank_cross_binary",   "rank", True,  "binary", 10, 3),   # C1+C2+C3 — expect drift
    ("rank_cross_soft",     "rank", True,  "soft",   10, 3),   # C1+C2
    ("rank_diag_binary",    "rank", False, "binary", 10, 3),   # C1+C3
    ("rank_diag_soft",      "rank", False, "soft",   10, 3),   # C1 only
    ("mse_cross_binary",    "mse",  True,  "binary", 10, 3),   # C2+C3
    ("mse_cross_soft",      "mse",  True,  "soft",   10, 3),   # C2 only
]


def run_ablation():
    results = load_results(RESULT_PATH)
    if "_cells" not in results:
        results["_cells"] = {}

    total_runs = len(CELLS) * N_SEEDS * len(VARIANTS)
    done = 0

    for cell_label, obj, cross, constr, d_dim, d_oh in CELLS:
        if cell_label not in results["_cells"]:
            results["_cells"][cell_label] = {}
        cell = results["_cells"][cell_label]

        log.info("\n--- Cell: %s (obj=%s, cross=%s, constr=%s, d=%d) ---",
                 cell_label, obj, cross, constr, d_dim)

        for seed in SEEDS:
            problem = RankBarrierProblem(
                d=d_dim, d_overhead=d_oh, N=1000,
                objective=obj, constraint=constr,
                cross_terms=cross, tau=0.05, seed=seed,
            )

            for variant in VARIANTS:
                key = f"{variant}_seed{seed}"
                if key in cell:
                    done += 1
                    continue

                trial = run_synthetic_trial(
                    problem, variant, seed, MAX_GEN, POP_SIZE)
                cell[key] = trial
                done += 1

                log.info("  [done] %s/%s s%d — regime=%s phi=%.3f (%.1fs) [%d/%d]",
                         cell_label, variant, seed,
                         trial["final"].get("regime", "?"),
                         trial["final"].get("phi_overhead", 0),
                         trial["elapsed_s"], done, total_runs)

                if done % 10 == 0:
                    save_results(results, RESULT_PATH)

        save_results(results, RESULT_PATH)

    _summarize_ablation(results)


def _summarize_ablation(results):
    log.info("\n" + "=" * 70)
    log.info("Synthetic ablation matrix summary")
    log.info("=" * 70)
    log.info("%-25s  %-8s  %-8s  %-8s  Fisher(full vs sep)",
             "Cell", "full", "sep", "iso")
    log.info("-" * 75)

    summary = {}

    for cell_label, obj, cross, constr, *_ in CELLS:
        cell = results["_cells"].get(cell_label, {})
        rates = {}
        for variant in VARIANTS:
            drift = sum(1 for s in SEEDS
                        if f"{variant}_seed{s}" in cell
                        and cell[f"{variant}_seed{s}"]["final"].get("regime") == "B")
            total = sum(1 for s in SEEDS if f"{variant}_seed{s}" in cell)
            rates[variant] = (drift, total)

        # Fisher's exact (full vs sep)
        fd, ft = rates["full"]
        sd, st = rates["sep"]
        p_str = "—"
        if ft > 0 and st > 0:
            table = [[fd, ft - fd], [sd, st - sd]]
            _, p = fisher_exact(table, alternative="greater")
            p_str = f"p={p:.4f}"

        log.info("%-25s  %2d/%-3d   %2d/%-3d   %2d/%-3d   %s",
                 cell_label,
                 rates["full"][0], rates["full"][1],
                 rates["sep"][0], rates["sep"][1],
                 rates["isotropic"][0], rates["isotropic"][1],
                 p_str)

        summary[cell_label] = {
            "conditions": {"C1": obj == "rank", "C2": cross, "C3": constr == "binary"},
            "rates": {v: {"drift": d, "total": t} for v, (d, t) in rates.items()},
            "fisher_p": p_str,
        }

    results["_summary"] = summary
    save_results(results, RESULT_PATH)


def show_status():
    results = load_results(RESULT_PATH)
    cells = results.get("_cells", {})
    for cell_label, *_rest in CELLS:
        cell = cells.get(cell_label, {})
        total = N_SEEDS * len(VARIANTS)
        done = len(cell)
        print(f"  {cell_label}: {done}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    else:
        run_ablation()


if __name__ == "__main__":
    main()
