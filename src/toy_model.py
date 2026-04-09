#!/usr/bin/env python3
"""
Minimal Deceptive Attractor (MDA) — Rotated Ellipsoid Model
=============================================================

Demonstrates that CMA-ES cross-dimensional covariance adaptation causes
the optimizer to follow rotated gradients that cross feasibility boundaries,
while diagonal (sep) and isotropic variants stay feasible.

Model
-----
d = d_signal + d_sensitivity dimensions.
R: random orthogonal rotation matrix (fixed by seed).
t: target vector [1,...,1,0,...,0] (signal components = 1, sensitivity = 0).

  objective:   f(theta) = sum_i c_i * (R @ theta - t)_i^2
  constraint:  ||theta[d_signal:]||_1 <= tau
  fitness:     f(theta) + P * max(0, ||theta[d_signal:]||_1 - tau)

The unconstrained optimum is theta* = R^T @ t, which has nonzero
sensitivity components due to rotation. Full CMA-ES discovers this
rotated path; sep-CMA-ES cannot.

Three CMA-ES variants compared:
  full:       standard CMA-ES (learns full d x d covariance)
  sep:        diagonal-only covariance (off-diagonal zeroed after update)
  isotropic:  C = I frozen, only sigma + mean adapt
"""

import numpy as np
import json
import os
import sys
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def make_problem(d=20, d_signal=15, condition_ratio=100.0, seed=0):
    """Create rotated ellipsoid problem components.

    Returns
    -------
    R : (d, d) orthogonal rotation matrix
    t : (d,)  target in rotated frame
    cond : (d,) condition numbers (log-spaced from 1 to condition_ratio)
    theta_star : (d,) unconstrained optimum in original frame
    """
    rng = np.random.RandomState(seed)

    # Haar-distributed random rotation
    H = rng.randn(d, d)
    R, _ = np.linalg.qr(H)

    # Target: only signal dims active in rotated frame
    t = np.zeros(d)
    t[:d_signal] = 1.0

    # Log-spaced condition numbers
    cond = np.logspace(0, np.log10(condition_ratio), d)

    # Unconstrained optimum in original frame
    theta_star = R.T @ t

    return R, t, cond, theta_star


def evaluate(theta_batch, R, t, cond, d_signal, tau, penalty):
    """Compute fitness for batch of candidates.

    fitness = rotated_ellipsoid(theta) + penalty * max(0, ||x||_1 - tau)
    """
    y = theta_batch @ R.T          # (pop, d) in rotated frame
    diff = y - t
    f = (diff ** 2 * cond).sum(axis=1)

    x_l1 = np.sum(np.abs(theta_batch[:, d_signal:]), axis=1)
    violation = np.maximum(0, x_l1 - tau)

    return f + penalty * violation


# ---------------------------------------------------------------------------
# Minimal CMA-ES (full / sep / isotropic)
# ---------------------------------------------------------------------------

class SimpleCMAES:
    """Minimal CMA-ES supporting full/sep/isotropic modes."""

    def __init__(self, d, sigma0=1.0, pop_size=None, seed=None,
                 sigma_min=1e-8, sigma_max=100.0):
        self.d = d
        self.sigma = sigma0
        self.sigma0 = sigma0
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mean = np.zeros(d)
        self.C = np.eye(d)
        self.pop_size = pop_size or max(14, 4 + int(3 * np.log(d)))
        self.mu = self.pop_size // 2
        self.rng = np.random.RandomState(seed)

        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0, np.sqrt((self.mu_eff - 1)
                        / (d + 1)) - 1) + self.c_sigma)
        self.cc = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        self.c1 = 2 / ((d + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                       / ((d + 2) ** 2 + self.mu_eff))
        self.chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        self.p_sigma = np.zeros(d)
        self.p_c = np.zeros(d)
        self.gen = 0
        self.n_restarts = 0

        self._evals = np.ones(d)
        self._evecs = np.eye(d)
        self._stale = True

    def _decompose(self):
        if self._stale:
            self.C = (self.C + self.C.T) / 2
            evals, evecs = np.linalg.eigh(self.C)
            evals = np.maximum(evals, 1e-20)
            cond = evals.max() / evals.min()
            if cond > 1e14:
                self.C = np.eye(self.d)
                evals = np.ones(self.d)
                evecs = np.eye(self.d)
            self._evals = evals
            self._evecs = evecs
            self._stale = False

    def ask(self):
        self._decompose()
        D_sqrt = np.sqrt(self._evals)
        z = self.rng.randn(self.pop_size, self.d)
        return self.mean + self.sigma * (z * D_sqrt) @ self._evecs.T

    def tell(self, candidates, fitness, mode="full"):
        idx = np.argsort(fitness)
        sel = candidates[idx[:self.mu]]

        old_mean = self.mean.copy()
        self.mean = self.weights @ sel
        dm = (self.mean - old_mean) / self.sigma

        self._decompose()
        inv_sqrt_D = 1.0 / np.sqrt(self._evals)
        C_invsqrt = (self._evecs * inv_sqrt_D) @ self._evecs.T

        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma)
                                  * self.mu_eff) * (C_invsqrt @ dm))

        ps_norm = np.linalg.norm(self.p_sigma)
        gen_factor = 1 - (1 - self.c_sigma) ** (2 * (self.gen + 1))
        h_sigma = int(ps_norm / np.sqrt(max(gen_factor, 1e-20))
                      < (1.4 + 2 / (self.d + 1)) * self.chi_n)

        self.p_c = ((1 - self.cc) * self.p_c
                    + h_sigma * np.sqrt(self.cc * (2 - self.cc)
                                        * self.mu_eff) * dm)

        artmp = (sel - old_mean) / self.sigma
        rank_mu = (artmp.T * self.weights) @ artmp
        c1_adj = (1 - h_sigma) * self.cc * (2 - self.cc)

        self.C = ((1 - self.c1 - self.cmu + c1_adj * self.c1) * self.C
                  + self.c1 * np.outer(self.p_c, self.p_c)
                  + self.cmu * rank_mu)

        if mode == "sep":
            self.C = np.diag(np.diag(self.C))
        elif mode == "isotropic":
            self.C = np.eye(self.d)

        self._stale = True

        self.sigma *= np.exp(
            self.c_sigma / self.d_sigma * (ps_norm / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
        self.gen += 1

    def should_restart(self):
        return self.sigma < self.sigma_min * 10

    def restart(self, mode="full"):
        """Restart sigma and paths, keep mean."""
        self.n_restarts += 1
        self.sigma = self.sigma0
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)
        self.C = np.eye(self.d)
        self._stale = True


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_comparison(d=20, d_signal=15, condition_ratio=100.0, tau=0.3,
                   penalty=10.0, max_gen=5000, pop_size=50,
                   seeds=None, problem_seed=0):
    """Run full/sep/isotropic comparison on rotated ellipsoid."""
    if seeds is None:
        seeds = [42, 137, 256, 512, 1024]

    d_sens = d - d_signal
    R, t, cond, theta_star = make_problem(
        d=d, d_signal=d_signal, condition_ratio=condition_ratio,
        seed=problem_seed,
    )

    x_star_l1 = float(np.sum(np.abs(theta_star[d_signal:])))
    f_star = 0.0  # minimum of unconstrained problem
    log.info("Problem: d=%d [%d+%d], cond_ratio=%.0f, tau=%.2f, P=%.1f",
             d, d_signal, d_sens, condition_ratio, tau, penalty)
    log.info("  theta* sensitivity ||x||_1 = %.3f (tau=%.2f, %s)",
             x_star_l1, tau,
             "INFEASIBLE" if x_star_l1 > tau else "feasible")

    conditions = ["full", "sep", "isotropic"]
    all_results = {}

    for mode in conditions:
        log.info("=" * 60)
        log.info("Condition: %s", mode)
        log.info("=" * 60)

        cond_results = []

        for seed in seeds:
            es = SimpleCMAES(d, sigma0=2.0, pop_size=pop_size, seed=seed)

            best_genome = None
            best_f = float("inf")
            trajectory = []

            for gen in range(max_gen):
                candidates = es.ask()
                fitness = evaluate(candidates, R, t, cond,
                                   d_signal, tau, penalty)
                es.tell(candidates, fitness, mode=mode)

                ib = fitness.argmin()
                if fitness[ib] < best_f:
                    best_f = float(fitness[ib])
                    best_genome = candidates[ib].copy()

                if es.should_restart():
                    es.restart(mode=mode)

                if (gen + 1) % 100 == 0:
                    x_l1 = float(np.sum(np.abs(best_genome[d_signal:])))
                    regime = "C" if x_l1 <= tau else "B"
                    # Raw objective (without penalty)
                    y = best_genome @ R.T
                    raw_f = float(((y - t) ** 2 * cond).sum())
                    trajectory.append({
                        "gen": gen + 1,
                        "raw_f": raw_f,
                        "fitness": best_f,
                        "regime": regime,
                        "x_l1": x_l1,
                        "feasible": x_l1 <= tau,
                        "sigma": float(es.sigma),
                    })

                if (gen + 1) % 1000 == 0:
                    x_l1 = float(np.sum(np.abs(best_genome[d_signal:])))
                    regime = "C" if x_l1 <= tau else "B"
                    y = best_genome @ R.T
                    raw_f = float(((y - t) ** 2 * cond).sum())
                    log.info("  %s seed %d gen %4d | raw_f=%.4f | "
                             "x_l1=%.3f | %s | sigma=%.2e",
                             mode, seed, gen + 1, raw_f, x_l1,
                             regime, es.sigma)

            x_l1_final = float(np.sum(np.abs(best_genome[d_signal:])))
            regime_final = "C" if x_l1_final <= tau else "B"
            y = best_genome @ R.T
            raw_f_final = float(((y - t) ** 2 * cond).sum())

            final = {
                "raw_f": raw_f_final,
                "fitness": best_f,
                "regime": regime_final,
                "x_l1": x_l1_final,
                "feasible": x_l1_final <= tau,
            }

            cond_results.append({
                "seed": seed, "final": final, "trajectory": trajectory,
            })
            log.info("  %s seed %d FINAL: raw_f=%.4f regime=%s "
                     "x_l1=%.3f feas=%s",
                     mode, seed, raw_f_final, regime_final,
                     x_l1_final, final["feasible"])

        all_results[mode] = cond_results

        regimes = {"B": 0, "C": 0}
        for cr in cond_results:
            regimes[cr["final"]["regime"]] += 1
        log.info("  %s regime distribution: B=%d C=%d",
                 mode, regimes["B"], regimes["C"])

    return all_results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_results(results, tau, out_path):
    """Two-panel figure: sensitivity drift + raw objective."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colours = {"full": "#d62728", "sep": "#2ca02c", "isotropic": "#1f77b4"}
    labels = {"full": "Full CMA-ES", "sep": "sep-CMA-ES",
              "isotropic": "Isotropic"}

    for mode in ["full", "sep", "isotropic"]:
        all_traj = results[mode]
        gens = [t["gen"] for t in all_traj[0]["trajectory"]]

        x_l1_matrix = np.array([
            [t["x_l1"] for t in cr["trajectory"]]
            for cr in all_traj
        ])
        rawf_matrix = np.array([
            [t["raw_f"] for t in cr["trajectory"]]
            for cr in all_traj
        ])

        mean_x = x_l1_matrix.mean(axis=0)
        std_x = x_l1_matrix.std(axis=0)
        mean_f = rawf_matrix.mean(axis=0)
        std_f = rawf_matrix.std(axis=0)

        c = colours[mode]
        ax1.plot(gens, mean_x, color=c, label=labels[mode], linewidth=1.5)
        ax1.fill_between(gens, mean_x - std_x, mean_x + std_x,
                         color=c, alpha=0.15)

        ax2.plot(gens, mean_f, color=c, label=labels[mode], linewidth=1.5)
        ax2.fill_between(gens, mean_f - std_f, mean_f + std_f,
                         color=c, alpha=0.15)

    ax1.axhline(tau, color="black", linestyle="--", linewidth=0.8,
                label=r"$\tau$ (feasibility)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel(r"Sensitivity magnitude $\|\theta_x\|_1$")
    ax1.set_title("Sensitivity drift")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Raw objective $f(\\theta)$")
    ax2.set_title("Objective quality (lower = better)")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--d-signal", type=int, default=15)
    p.add_argument("--condition-ratio", type=float, default=100.0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--penalty", type=float, default=10.0)
    p.add_argument("--max-gen", type=int, default=5000)
    p.add_argument("--pop-size", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 137, 256, 512, 1024])
    p.add_argument("--problem-seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    results = run_comparison(
        d=args.d, d_signal=args.d_signal,
        condition_ratio=args.condition_ratio,
        tau=args.tau, penalty=args.penalty,
        max_gen=args.max_gen, pop_size=args.pop_size,
        seeds=args.seeds, problem_seed=args.problem_seed,
    )

    out_dir = os.path.join(PROJECT_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "toy_model_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", json_path)

    if not args.no_plot:
        fig_path = os.path.join(PROJECT_DIR, "figures",
                                "toy_model_drift.pdf")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plot_results(results, args.tau, fig_path)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    for mode in ["full", "sep", "isotropic"]:
        regimes = {"B": 0, "C": 0}
        for cr in results[mode]:
            regimes[cr["final"]["regime"]] += 1
        f_avg = np.mean([cr["final"]["raw_f"] for cr in results[mode]])
        x_avg = np.mean([cr["final"]["x_l1"] for cr in results[mode]])
        log.info("  %-10s B=%d C=%d  raw_f_avg=%.4f  x_l1_avg=%.3f",
                 mode, regimes["B"], regimes["C"], f_avg, x_avg)


if __name__ == "__main__":
    main()
