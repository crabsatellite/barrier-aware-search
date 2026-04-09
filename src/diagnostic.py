"""
Diagnostic experiments to determine the paper's main conclusion:
  A) Single-feature baselines — does the composite measure actually beat individuals?
  B) Subset stability — is the train/test gap inherent or distributional noise?

These two results decide whether the narrative is:
  "combinatorial advantage found" vs "barrier-constrained instability discovered"
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "diagnostic.log")),
    ],
)
log = logging.getLogger(__name__)


def load_config():
    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        return json.load(f)


def load_search_data():
    return np.load(os.path.join(PROJECT_DIR, "data", "search_features.npz"))


def load_best_genome():
    with open(os.path.join(PROJECT_DIR, "data", "top_candidates.json")) as f:
        results = json.load(f)
    return np.array(results["candidates"][0]["genome"])


# ═══════════════════════════════════════════════════════════════════════════
# A) Single-feature baselines
# ═══════════════════════════════════════════════════════════════════════════

def single_feature_baselines(measures, targets, names):
    """Spearman r for each individual measure vs circuit complexity."""
    log.info("=== Single-feature baselines ===")
    results = {}
    for i, name in enumerate(names):
        r, p = spearmanr(measures[:, i], targets)
        results[name] = {"r": float(r), "p": float(p)}
        log.info("  %-25s r = %.4f  (p = %.2e)", name, r, p)

    best_name = max(results, key=lambda k: abs(results[k]["r"]))
    best_r = results[best_name]["r"]
    log.info("  Best single feature: %s (r = %.4f)", best_name, best_r)
    return results, best_name, best_r


# ═══════════════════════════════════════════════════════════════════════════
# B) Subset stability
# ═══════════════════════════════════════════════════════════════════════════

def subset_stability(genome, measures, quad, targets, n_subsets=50,
                     fractions=(0.05, 0.10, 0.15, 0.30, 0.50, 0.70, 0.90)):
    """Evaluate composite measure's Spearman r on random subsets of varying size."""
    from measures import MeasureEvaluator

    log.info("=== Subset stability (%d subsets × %d fractions) ===",
             n_subsets, len(fractions))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(targets)

    # evaluate mu on full dataset once
    ev = MeasureEvaluator(measures, quad, device)
    mu_full = ev.evaluate_single(genome)

    # full-dataset correlation
    r_full, _ = spearmanr(mu_full, targets)
    log.info("  Full dataset (N=%d): r = %.4f", N, r_full)

    results = {"full_r": float(r_full), "N": N, "fractions": {}}

    rng = np.random.RandomState(123)
    for frac in fractions:
        size = int(N * frac)
        corrs = []
        for _ in range(n_subsets):
            idx = rng.choice(N, size=size, replace=False)
            r, _ = spearmanr(mu_full[idx], targets[idx])
            corrs.append(float(r))

        arr = np.array(corrs)
        entry = {
            "size": size,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "values": corrs,
        }
        results["fractions"][str(frac)] = entry
        log.info("  frac=%.2f (N=%6d): mean=%.4f  std=%.4f  [%.4f, %.4f]",
                 frac, size, arr.mean(), arr.std(), arr.min(), arr.max())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# C) Per-circuit-size breakdown
# ═══════════════════════════════════════════════════════════════════════════

def per_size_analysis(genome, measures, quad, targets):
    """How well does the measure rank functions WITHIN each circuit size bucket?"""
    from measures import MeasureEvaluator

    log.info("=== Per-circuit-size analysis ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MeasureEvaluator(measures, quad, device)
    mu_full = ev.evaluate_single(genome)

    sizes = np.unique(targets.astype(int))
    counts = {}
    for s in sizes:
        mask = targets.astype(int) == s
        n = mask.sum()
        mu_s = mu_full[mask]
        counts[int(s)] = {
            "count": int(n),
            "mu_mean": float(mu_s.mean()),
            "mu_std": float(mu_s.std()),
        }
        log.info("  size=%2d: N=%6d  mu_mean=%+.4f  mu_std=%.4f",
                 s, n, mu_s.mean(), mu_s.std())

    # separation quality: can mu distinguish adjacent sizes?
    log.info("  --- Adjacent-size separation ---")
    separations = {}
    for i in range(len(sizes) - 1):
        s1, s2 = int(sizes[i]), int(sizes[i + 1])
        m1 = mu_full[targets.astype(int) == s1]
        m2 = mu_full[targets.astype(int) == s2]
        # Cohen's d
        pooled_std = np.sqrt((m1.var() + m2.var()) / 2)
        d = (m2.mean() - m1.mean()) / max(pooled_std, 1e-12)
        separations[f"{s1}-{s2}"] = float(d)
        log.info("  size %d→%d: Cohen's d = %.3f", s1, s2, d)

    return {"per_size": counts, "separations": separations}


def main():
    cfg = load_config()
    names = cfg["data"]["measure_names"]
    data = load_search_data()

    measures = data["knuth_measures"]
    targets = data["knuth_targets"]
    quad = data["knuth_quad"]

    genome = load_best_genome()

    # A
    baseline_results, best_single, best_single_r = single_feature_baselines(
        measures, targets, names)

    # B
    stability_results = subset_stability(genome, measures, quad, targets)

    # C
    size_results = per_size_analysis(genome, measures, quad, targets)

    # composite vs best single
    composite_r = stability_results["full_r"]
    improvement = composite_r - best_single_r
    log.info("\n=== Summary ===")
    log.info("  Best single feature: %s (r=%.4f)", best_single, best_single_r)
    log.info("  Composite measure:   r=%.4f", composite_r)
    log.info("  Improvement:         +%.4f (%.1f%%)",
             improvement, 100 * improvement / abs(best_single_r))
    log.info("  Subset stability at 15%%: mean=%.4f  std=%.4f",
             stability_results["fractions"]["0.15"]["mean"],
             stability_results["fractions"]["0.15"]["std"])

    # save
    out = {
        "baselines": baseline_results,
        "best_single_feature": best_single,
        "best_single_r": best_single_r,
        "composite_full_r": composite_r,
        "improvement_over_best_single": improvement,
        "subset_stability": stability_results,
        "per_size": size_results,
    }
    out_path = os.path.join(PROJECT_DIR, "data", "diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
