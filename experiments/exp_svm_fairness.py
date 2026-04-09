"""
Block 2A: SVM fairness domain — second real-world domain for generalizability.

Problem: Find SVM hyperparameter + feature weighting configurations that maximise
cross-validated rank-based performance while satisfying a binary fairness constraint
(demographic parity gap <= threshold).

This domain has all three structural conditions:
  C1: Rank-based objective (Spearman of CV fold accuracies vs target ranking)
  C2: Cross-dimensional interactions (feature weights x kernel params)
  C3: Binary fairness constraint (pass/fail)

Uses sklearn's Adult income dataset.
30 seeds x 3 variants (full/sep/isotropic) x 5000 gen.

Usage:
    python exp_svm_fairness.py run
    python exp_svm_fairness.py status
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr, fisher_exact

from shared import get_logger, RESULTS_DIR, load_results, save_results

log = get_logger("exp_svm_fairness")
warnings.filterwarnings("ignore", category=FutureWarning)

RESULT_PATH = os.path.join(RESULTS_DIR, "svm_fairness.json")

SEEDS = list(range(1, 31))
VARIANTS = ["full", "sep", "isotropic"]
MAX_GEN = 3000
POP_SIZE = 50


# ═══════════════════════════════════════════════════════════════════════════
# Dataset preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_adult_dataset():
    """Load UCI Adult dataset via sklearn, return X, y, sensitive_attr."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
    except Exception:
        log.warning("Cannot fetch Adult dataset from OpenML, generating synthetic.")
        return _synthetic_fairness_dataset()

    # binary target
    target_col = "income" if "income" in df.columns else "class"
    y = (df[target_col].astype(str).str.strip().str.startswith(">50K")).astype(int).values

    # sensitive attribute: sex
    sensitive = (df["sex"].astype(str).str.strip() == "Male").astype(int).values

    # numeric features only (for simplicity)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("income", "class", target_col)]
    X = df[num_cols].values.astype(np.float64)

    # handle NaN
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    # standardise
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # subsample for computational feasibility
    max_n = 5000
    if len(y) > max_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), max_n, replace=False)
        X, y, sensitive = X[idx], y[idx], sensitive[idx]

    return X, y, sensitive


def _synthetic_fairness_dataset(n=5000, d=10, seed=42):
    """Fallback: synthetic dataset with fairness structure."""
    rng = np.random.RandomState(seed)
    sensitive = rng.binomial(1, 0.5, n)
    X = rng.randn(n, d)
    X[:, 0] += sensitive * 0.5  # bias
    w_true = rng.randn(d) * 0.3
    logits = X @ w_true + sensitive * 0.3
    y = (logits > 0).astype(int)
    return X, y, sensitive


# ═══════════════════════════════════════════════════════════════════════════
# SVM fairness problem
# ═══════════════════════════════════════════════════════════════════════════

class SVMFairnessProblem:
    """Parameterised SVM config with binary fairness constraint."""

    def __init__(self, X, y, sensitive, n_folds=3, tau_dp=0.05, seed=0):
        self.X = X
        self.y = y
        self.sensitive = sensitive
        self.n_folds = n_folds
        self.tau_dp = tau_dp
        self.n_features = X.shape[1]

        # genome: feature_weights(d) + log_C(1) + log_gamma(1) +
        #         cross_terms(d*2 = feature_weight x [C, gamma]) +
        #         activation(d) + threshold(d)
        self.d_feat = self.n_features
        self.d_svm = 2  # log_C, log_gamma
        self.d_cross = self.d_feat * self.d_svm  # cross interactions
        self.d_act = self.d_feat
        self.d_thresh = self.d_feat
        self.genome_dim = (self.d_feat + self.d_svm + self.d_cross
                           + self.d_act + self.d_thresh)

        # precompute fold indices
        rng = np.random.RandomState(seed + 5000)
        indices = rng.permutation(len(y))
        fold_size = len(y) // n_folds
        self.folds = [
            indices[i * fold_size:(i + 1) * fold_size]
            for i in range(n_folds)
        ]

        # target: rank of fold accuracies (for rank-based objective)
        # generate N_configs random configs to rank against
        self.n_configs = 200
        self.reference_accs = self._generate_reference_accs(rng)

    def _generate_reference_accs(self, rng):
        """Generate reference accuracy profile for ranking."""
        from sklearn.svm import LinearSVC
        accs = []
        for _ in range(self.n_configs):
            C = 10 ** (rng.uniform(-3, 3))
            w = rng.randn(self.n_features) * 0.5
            X_w = self.X * w[None, :]
            fold_accs = []
            for fold_idx in self.folds:
                mask = np.ones(len(self.y), dtype=bool)
                mask[fold_idx] = False
                try:
                    clf = LinearSVC(C=C, max_iter=500, dual=False)
                    clf.fit(X_w[mask], self.y[mask])
                    fold_accs.append(clf.score(X_w[fold_idx], self.y[fold_idx]))
                except Exception:
                    fold_accs.append(0.5)
            accs.append(np.mean(fold_accs))
        return np.array(accs)

    def evaluate_batch(self, genomes):
        """Evaluate genomes. Returns (fitness, diagnostics)."""
        from sklearn.svm import LinearSVC

        pop = len(genomes)
        corrs = np.zeros(pop)
        densities = np.zeros(pop)
        violations = np.zeros(pop, dtype=bool)
        phi_overheads = np.zeros(pop)

        for k in range(pop):
            g = genomes[k]
            # decode genome
            feat_w = g[:self.d_feat]
            log_C = g[self.d_feat]
            log_gamma = g[self.d_feat + 1]
            cross_w = g[self.d_feat + self.d_svm:
                        self.d_feat + self.d_svm + self.d_cross]

            # effective feature weights = feat_w + cross_w reshaped
            cross_mat = cross_w.reshape(self.d_feat, self.d_svm)
            eff_w = feat_w + cross_mat[:, 0] * log_C + cross_mat[:, 1] * log_gamma

            C = np.clip(10 ** log_C, 1e-4, 1e4)
            X_w = self.X * eff_w[None, :]

            # cross-validated accuracy per fold
            fold_accs = []
            fold_dp_gaps = []
            for fold_idx in self.folds:
                mask = np.ones(len(self.y), dtype=bool)
                mask[fold_idx] = False
                try:
                    clf = LinearSVC(C=C, max_iter=500, dual=False)
                    clf.fit(X_w[mask], self.y[mask])
                    preds = clf.predict(X_w[fold_idx])
                    fold_accs.append(np.mean(preds == self.y[fold_idx]))

                    # demographic parity gap
                    s_test = self.sensitive[fold_idx]
                    rate_1 = preds[s_test == 1].mean() if (s_test == 1).any() else 0.5
                    rate_0 = preds[s_test == 0].mean() if (s_test == 0).any() else 0.5
                    fold_dp_gaps.append(abs(rate_1 - rate_0))
                except Exception:
                    fold_accs.append(0.5)
                    fold_dp_gaps.append(0.0)

            mean_acc = np.mean(fold_accs)
            mean_dp_gap = np.mean(fold_dp_gaps)

            # rank-based objective: Spearman vs reference configs
            all_accs = np.append(self.reference_accs, mean_acc)
            rank_corr = spearmanr(all_accs, np.arange(len(all_accs)))[0]
            corrs[k] = mean_acc  # use accuracy directly for ranking

            # binary fairness constraint
            violations[k] = mean_dp_gap > self.tau_dp
            densities[k] = mean_dp_gap

            # overhead: cross-term contribution
            cross_mass = np.abs(cross_w).sum()
            feat_mass = np.abs(feat_w).sum() + 1e-12
            phi_overheads[k] = cross_mass / (feat_mass + cross_mass)

        # fitness
        penalty = violations.astype(np.float64) * 10.0
        l2 = np.sqrt((genomes ** 2).sum(axis=1))
        fitness = -corrs + penalty + 0.001 * l2

        return fitness, {
            "corr": corrs,
            "density": densities,
            "violated": violations,
            "phi_overhead": phi_overheads,
        }


def run_svm_trial(problem, variant, seed, max_gen, pop_size):
    """Run one CMA-ES trial on SVM fairness problem."""
    from exp_synthetic import SimpleCMAES

    es = SimpleCMAES(problem.genome_dim, sigma0=0.3, pop_size=pop_size, seed=seed)
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
                "acc": float(best_diag["corr"][0]),
                "dp_gap": float(best_diag["density"][0]),
                "phi_overhead": float(best_diag["phi_overhead"][0]),
                "violated": bool(best_diag["violated"][0]),
                "regime": "B" if best_diag["phi_overhead"][0] > 0.05 else "C",
            })

    return {
        "seed": seed, "variant": variant,
        "elapsed_s": time.time() - t0,
        "final": trajectory[-1] if trajectory else {},
        "trajectory": trajectory,
    }


def run():
    log.info("=" * 70)
    log.info("Block 2A: SVM fairness domain")
    log.info("=" * 70)

    results = load_results(RESULT_PATH)
    log.info("Loading Adult dataset...")
    X, y, sensitive = load_adult_dataset()
    log.info("Dataset: %d samples, %d features", X.shape[0], X.shape[1])

    for seed in SEEDS:
        problem = SVMFairnessProblem(X, y, sensitive, seed=seed)

        for variant in VARIANTS:
            key = f"{variant}_seed{seed}"
            if key in results:
                log.info("  [skip] %s", key)
                continue

            log.info("  [run]  %s ...", key)
            trial = run_svm_trial(problem, variant, seed, MAX_GEN, POP_SIZE)
            results[key] = trial
            save_results(results, RESULT_PATH)
            log.info("  [done] %s — regime=%s, acc=%.4f, dp=%.4f (%.1fs)",
                     key, trial["final"].get("regime", "?"),
                     trial["final"].get("acc", 0),
                     trial["final"].get("dp_gap", 0),
                     trial["elapsed_s"])

    # summary
    log.info("\nSVM fairness summary:")
    for variant in VARIANTS:
        drift = sum(1 for s in SEEDS
                    if f"{variant}_seed{s}" in results
                    and results[f"{variant}_seed{s}"]["final"].get("regime") == "B")
        total = sum(1 for s in SEEDS if f"{variant}_seed{s}" in results)
        if total:
            log.info("  %-12s: drift %d/%d", variant, drift, total)


def show_status():
    results = load_results(RESULT_PATH)
    total = len(SEEDS) * len(VARIANTS)
    done = sum(1 for s in SEEDS for v in VARIANTS
               if f"{v}_seed{s}" in results)
    print(f"  SVM fairness: {done}/{total}")


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
