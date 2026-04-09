"""
Block 1A+1B+1C: 30-seed cross-optimizer comparison + 20-seed landscape.

Usage:
    python exp_30seed.py crossopt          # 30 seeds x 3 variants (Block 1A)
    python exp_30seed.py landscape         # 20 seeds, full CMA-ES (Block 1B)
    python exp_30seed.py all               # both
    python exp_30seed.py status            # show progress

Checkpoint-resumable: skips completed (seed, variant) pairs.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

from shared import (
    get_logger, RESULTS_DIR, load_results, save_results,
    result_key, setup_boolean_domain, run_cmaes_trial,
)

log = get_logger("exp_30seed")

CROSSOPT_PATH = os.path.join(RESULTS_DIR, "crossopt_30seed.json")
LANDSCAPE_PATH = os.path.join(RESULTS_DIR, "landscape_20seed.json")

MAX_GEN = 3000
POP_SIZE = 50
CHECKPOINTS = [500, 1000, 2000, 3000]
CROSSOPT_SEEDS = list(range(1, 31))
LANDSCAPE_SEEDS = list(range(1, 21))
VARIANTS = ["full", "sep", "isotropic"]


def run_crossopt():
    """Block 1A: 30-seed cross-optimizer comparison."""
    log.info("=" * 70)
    log.info("Block 1A: 30-seed cross-optimizer comparison")
    log.info("=" * 70)

    results = load_results(CROSSOPT_PATH)
    domain = setup_boolean_domain()

    total = len(CROSSOPT_SEEDS) * len(VARIANTS)
    done = sum(1 for s in CROSSOPT_SEEDS for v in VARIANTS
               if result_key(v, s) in results)
    log.info("Progress: %d / %d completed", done, total)

    for seed in CROSSOPT_SEEDS:
        for variant in VARIANTS:
            key = result_key(variant, seed)
            if key in results:
                log.info("  [skip] %s (already done)", key)
                continue

            log.info("  [run]  %s ...", key)
            trial = run_cmaes_trial(
                seed=seed, variant=variant,
                max_gen=MAX_GEN, pop_size=POP_SIZE,
                domain=domain, checkpoints=CHECKPOINTS,
            )
            results[key] = trial
            save_results(results, CROSSOPT_PATH)
            log.info("  [done] %s — regime=%s, r=%.4f, phi_sens=%.4f (%.1fs)",
                     key, trial["final"]["regime"], trial["final"]["r"],
                     trial["final"]["phi_sens"], trial["elapsed_s"])

    # summary statistics
    _summarize_crossopt(results)


def run_landscape():
    """Block 1B: 20-seed landscape characterization under full CMA-ES."""
    log.info("=" * 70)
    log.info("Block 1B: 20-seed landscape characterization")
    log.info("=" * 70)

    results = load_results(LANDSCAPE_PATH)
    domain = setup_boolean_domain()

    for seed in LANDSCAPE_SEEDS:
        key = result_key("full", seed)
        if key in results:
            log.info("  [skip] %s", key)
            continue

        log.info("  [run]  %s ...", key)
        trial = run_cmaes_trial(
            seed=seed, variant="full",
            max_gen=MAX_GEN, pop_size=POP_SIZE,
            domain=domain, checkpoints=CHECKPOINTS,
        )
        results[key] = trial
        save_results(results, LANDSCAPE_PATH)
        log.info("  [done] %s — regime=%s", key, trial["final"]["regime"])

    _summarize_landscape(results)


def _summarize_crossopt(results):
    log.info("\n" + "=" * 70)
    log.info("Cross-optimizer summary (30 seeds)")
    log.info("=" * 70)

    for variant in VARIANTS:
        regimes = []
        for seed in CROSSOPT_SEEDS:
            key = result_key(variant, seed)
            if key in results:
                regimes.append(results[key]["final"]["regime"])
        if not regimes:
            continue

        n = len(regimes)
        drift = sum(1 for r in regimes if r in ("A", "B"))
        log.info("  %-12s: drift %d/%d (%.0f%%)  A=%d B=%d C=%d",
                 variant, drift, n, 100 * drift / n,
                 regimes.count("A"), regimes.count("B"), regimes.count("C"))

    # Fisher's exact test: full vs sep
    full_drift = sum(1 for s in CROSSOPT_SEEDS
                     if result_key("full", s) in results
                     and results[result_key("full", s)]["final"]["regime"] in ("A", "B"))
    full_no = sum(1 for s in CROSSOPT_SEEDS
                  if result_key("full", s) in results
                  and results[result_key("full", s)]["final"]["regime"] == "C")
    sep_drift = sum(1 for s in CROSSOPT_SEEDS
                    if result_key("sep", s) in results
                    and results[result_key("sep", s)]["final"]["regime"] in ("A", "B"))
    sep_no = sum(1 for s in CROSSOPT_SEEDS
                 if result_key("sep", s) in results
                 and results[result_key("sep", s)]["final"]["regime"] == "C")

    if full_drift + full_no > 0 and sep_drift + sep_no > 0:
        table = [[full_drift, full_no], [sep_drift, sep_no]]
        odds, p_two = fisher_exact(table)
        _, p_one = fisher_exact(table, alternative="greater")
        log.info("\n  Fisher's exact (full vs sep):")
        log.info("    two-sided p = %.4f", p_two)
        log.info("    one-sided p = %.4f  (full > sep)", p_one)
        log.info("    odds ratio  = %.2f", odds)

        # save summary
        results["_summary"] = {
            "full_drift": full_drift, "full_total": full_drift + full_no,
            "sep_drift": sep_drift, "sep_total": sep_drift + sep_no,
            "fisher_p_two": p_two, "fisher_p_one": p_one,
            "odds_ratio": odds,
        }
        save_results(results, CROSSOPT_PATH)


def _summarize_landscape(results):
    regimes = []
    for seed in LANDSCAPE_SEEDS:
        key = result_key("full", seed)
        if key in results:
            regimes.append(results[key]["final"]["regime"])
    if regimes:
        log.info("\n  Landscape (20 seeds, full CMA-ES, 8000 gen):")
        log.info("    A=%d  B=%d  C=%d",
                 regimes.count("A"), regimes.count("B"), regimes.count("C"))


def show_status():
    for label, path, seeds in [
        ("Crossopt", CROSSOPT_PATH, CROSSOPT_SEEDS),
        ("Landscape", LANDSCAPE_PATH, LANDSCAPE_SEEDS),
    ]:
        results = load_results(path)
        variants = VARIANTS if "crossopt" in path.lower() else ["full"]
        total = len(seeds) * len(variants)
        done = sum(1 for s in seeds for v in variants
                   if result_key(v, s) in results)
        print(f"{label}: {done}/{total} completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["crossopt", "landscape", "all", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    elif args.mode == "crossopt":
        run_crossopt()
    elif args.mode == "landscape":
        run_landscape()
    elif args.mode == "all":
        run_crossopt()
        run_landscape()


if __name__ == "__main__":
    main()
