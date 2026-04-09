"""
Curvature Correction Hypothesis — test whether calibration = curvature correction on signal embedding.

Experiments:
  curvature_test      — signal-only vs full: show flattening in mid-range sizes
  artificial_correct   — replace sensitivity with mu_signal^2, optimize alpha
  rank_density         — per-size rank overlap for signal vs full
  all                  — run all three

Usage: python curvature.py <experiment>
"""

import json
import logging
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "curvature.log")),
    ],
)
log = logging.getLogger(__name__)

from mechanism import (
    MEASURE_NAMES, D, Q_DIM, GENOME_DIM, W_SLICE, Q_SLICE, A_SLICE, T_SLICE,
    load_data, load_best_genome,
)
from measures import MeasureEvaluator

RESULT_PATH = os.path.join(PROJECT_DIR, "data", "curvature_results.json")


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {}


def save_results(data):
    with open(RESULT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def evaluate_genome_values(genome, measures, quad):
    """Evaluate genome, return per-function mu values (not just correlation)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MeasureEvaluator(measures, quad, device)
    return ev.evaluate_single(genome)


def zero_feature(genome, fi):
    """Zero all genome parameters related to feature fi."""
    g = genome.copy()
    g[fi] = 0.0  # linear
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == fi or j == fi:
                g[D + k] = 0.0
            k += 1
    g[D + Q_DIM + fi] = 0.0      # threshold activation
    g[D + Q_DIM + D + fi] = 0.0  # threshold position
    return g


def keep_only_features(genome, feature_indices):
    """Keep only specified features, zero everything else."""
    g = np.zeros_like(genome)
    for fi in feature_indices:
        g[fi] = genome[fi]  # linear
        g[D + Q_DIM + fi] = genome[D + Q_DIM + fi]      # threshold activation
        g[D + Q_DIM + D + fi] = genome[D + Q_DIM + D + fi]  # threshold position
    # quadratic: keep only terms where BOTH features are in the set
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i in feature_indices and j in feature_indices:
                g[D + k] = genome[D + k]
            k += 1
    return g


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Curvature test
# ═══════════════════════════════════════════════════════════════════════════

def curvature_test(genome, measures, quad, targets):
    """Show that signal embedding has flattened mid-range, calibration unfolds it."""
    log.info("=" * 70)
    log.info("EXPERIMENT 1: CURVATURE TEST")
    log.info("=" * 70)

    si = MEASURE_NAMES.index("sensitivity")
    se = MEASURE_NAMES.index("spectral_entropy")
    inf_i = MEASURE_NAMES.index("influence")

    # Three measure variants
    # 1. Full measure (baseline)
    mu_full = evaluate_genome_values(genome, measures, quad)
    r_full, _ = spearmanr(mu_full, targets)

    # 2. Signal only (remove sensitivity = remove calibration)
    g_no_sens = zero_feature(genome, si)
    mu_no_sens = evaluate_genome_values(g_no_sens, measures, quad)
    r_no_sens, _ = spearmanr(mu_no_sens, targets)

    # 3. Pure signal axis (only spectral_entropy + influence)
    g_signal = keep_only_features(genome, [se, inf_i])
    mu_signal = evaluate_genome_values(g_signal, measures, quad)
    r_signal, _ = spearmanr(mu_signal, targets)

    log.info("  r_full      = %.4f  (all terms)", r_full)
    log.info("  r_no_sens   = %.4f  (sensitivity zeroed)", r_no_sens)
    log.info("  r_signal    = %.4f  (only spectral_entropy + influence)", r_signal)

    # Per-size analysis: rank density and rank spread
    sizes = np.unique(targets.astype(int))
    log.info("")
    log.info("  Per-size rank statistics:")
    log.info("  %4s  %6s  %12s %12s %12s  %12s %12s %12s",
             "size", "N",
             "mu_sig_mean", "mu_sig_std", "mu_sig_IQR",
             "mu_ful_mean", "mu_ful_std", "mu_ful_IQR")

    per_size = {}
    for s in sizes:
        mask = targets.astype(int) == s
        n = int(mask.sum())

        sig_vals = mu_signal[mask]
        ful_vals = mu_full[mask]
        nosens_vals = mu_no_sens[mask]

        per_size[int(s)] = {
            "n": n,
            "signal": {
                "mean": float(np.mean(sig_vals)),
                "std": float(np.std(sig_vals)),
                "iqr": float(np.percentile(sig_vals, 75) - np.percentile(sig_vals, 25)),
                "min": float(np.min(sig_vals)),
                "max": float(np.max(sig_vals)),
            },
            "no_sens": {
                "mean": float(np.mean(nosens_vals)),
                "std": float(np.std(nosens_vals)),
                "iqr": float(np.percentile(nosens_vals, 75) - np.percentile(nosens_vals, 25)),
            },
            "full": {
                "mean": float(np.mean(ful_vals)),
                "std": float(np.std(ful_vals)),
                "iqr": float(np.percentile(ful_vals, 75) - np.percentile(ful_vals, 25)),
                "min": float(np.min(ful_vals)),
                "max": float(np.max(ful_vals)),
            },
        }

        log.info("  %4d  %6d  %12.4f %12.4f %12.4f  %12.4f %12.4f %12.4f",
                 s, n,
                 np.mean(sig_vals), np.std(sig_vals),
                 np.percentile(sig_vals, 75) - np.percentile(sig_vals, 25),
                 np.mean(ful_vals), np.std(ful_vals),
                 np.percentile(ful_vals, 75) - np.percentile(ful_vals, 25))

    # Rank overlap analysis: for adjacent sizes, what fraction of rank ranges overlap?
    log.info("")
    log.info("  Rank overlap between adjacent sizes:")
    log.info("  %8s  %10s  %10s  %10s", "sizes", "sig_olap", "nosens_olap", "full_olap")

    # Compute global ranks
    rank_signal = rankdata(mu_signal, method='average')
    rank_full = rankdata(mu_full, method='average')
    rank_nosens = rankdata(mu_no_sens, method='average')

    overlap_data = {}
    for idx in range(len(sizes) - 1):
        s1, s2 = int(sizes[idx]), int(sizes[idx + 1])
        m1 = targets.astype(int) == s1
        m2 = targets.astype(int) == s2

        def _overlap(ranks, mask1, mask2):
            r1 = ranks[mask1]
            r2 = ranks[mask2]
            lo = max(np.percentile(r1, 25), np.percentile(r2, 25))
            hi = min(np.percentile(r1, 75), np.percentile(r2, 75))
            if hi <= lo:
                return 0.0
            iqr1 = np.percentile(r1, 75) - np.percentile(r1, 25)
            iqr2 = np.percentile(r2, 75) - np.percentile(r2, 25)
            avg_iqr = (iqr1 + iqr2) / 2
            return float((hi - lo) / avg_iqr) if avg_iqr > 0 else 0.0

        o_sig = _overlap(rank_signal, m1, m2)
        o_nosens = _overlap(rank_nosens, m1, m2)
        o_ful = _overlap(rank_full, m1, m2)

        overlap_data[f"{s1}-{s2}"] = {
            "signal_overlap": o_sig,
            "no_sens_overlap": o_nosens,
            "full_overlap": o_ful,
        }
        log.info("  %4d-%2d  %10.3f  %10.3f  %10.3f", s1, s2, o_sig, o_nosens, o_ful)

    # Generate curvature visualization
    _plot_curvature(per_size, sizes, mu_signal, mu_full, mu_no_sens, targets)

    return {
        "r_full": r_full, "r_no_sens": r_no_sens, "r_signal": r_signal,
        "per_size": per_size,
        "rank_overlap": overlap_data,
    }


def _plot_curvature(per_size, sizes, mu_signal, mu_full, mu_no_sens, targets):
    """Generate curvature visualization figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")
    sizes = sorted(per_size.keys())

    # Figure 1: Mean mu by circuit size (signal vs full)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: means with error bars
    ax = axes[0]
    sig_means = [per_size[s]["signal"]["mean"] for s in sizes]
    sig_stds = [per_size[s]["signal"]["std"] for s in sizes]
    ful_means = [per_size[s]["full"]["mean"] for s in sizes]
    ful_stds = [per_size[s]["full"]["std"] for s in sizes]
    nosens_means = [per_size[s]["no_sens"]["mean"] for s in sizes]

    ax.errorbar(sizes, sig_means, yerr=sig_stds, fmt='o-', color='#1976d2',
                label='Signal only', capsize=3, markersize=4)
    ax.errorbar(sizes, ful_means, yerr=ful_stds, fmt='s-', color='#d32f2f',
                label='Full (signal + calibration)', capsize=3, markersize=4)
    ax.plot(sizes, nosens_means, 'D--', color='#4caf50',
            label='No sensitivity', markersize=4)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Mean mu")
    ax.set_title("A. Embedding by circuit size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: IQR (spread) by size
    ax = axes[1]
    sig_iqrs = [per_size[s]["signal"]["iqr"] for s in sizes]
    ful_iqrs = [per_size[s]["full"]["iqr"] for s in sizes]
    nosens_iqrs = [per_size[s]["no_sens"]["iqr"] for s in sizes]

    ax.plot(sizes, sig_iqrs, 'o-', color='#1976d2', label='Signal only', markersize=4)
    ax.plot(sizes, ful_iqrs, 's-', color='#d32f2f', label='Full', markersize=4)
    ax.plot(sizes, nosens_iqrs, 'D--', color='#4caf50', label='No sensitivity', markersize=4)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("IQR of mu")
    ax.set_title("B. Intra-size spread (rank resolution)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: rank separation (mean rank per size)
    ax = axes[2]
    rank_signal = rankdata(mu_signal, method='average')
    rank_full = rankdata(mu_full, method='average')
    rank_nosens = rankdata(mu_no_sens, method='average')

    sig_rank_means = []
    ful_rank_means = []
    nosens_rank_means = []
    for s in sizes:
        mask = targets.astype(int) == s
        sig_rank_means.append(float(np.mean(rank_signal[mask])))
        ful_rank_means.append(float(np.mean(rank_full[mask])))
        nosens_rank_means.append(float(np.mean(rank_nosens[mask])))

    ax.plot(sizes, sig_rank_means, 'o-', color='#1976d2', label='Signal only', markersize=4)
    ax.plot(sizes, ful_rank_means, 's-', color='#d32f2f', label='Full', markersize=4)
    ax.plot(sizes, nosens_rank_means, 'D--', color='#4caf50', label='No sensitivity', markersize=4)
    # ideal: diagonal
    N = len(targets)
    ideal_ranks = []
    for s in sizes:
        mask = targets.astype(int) == s
        n = mask.sum()
        # ideal rank = position if perfectly sorted by size
        ideal_ranks.append(float(np.mean(rankdata(targets, method='average')[mask])))
    ax.plot(sizes, ideal_ranks, 'k--', alpha=0.5, label='Ideal (perfect ranking)', linewidth=1)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Mean rank")
    ax.set_title("C. Rank separation by size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Curvature correction: signal embedding vs calibrated measure", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "curvature_test.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved curvature_test.pdf")

    # Figure 2: Violin/box plot of mu distributions per size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    positions = list(range(len(sizes)))

    # Signal only
    sig_data = [mu_signal[targets.astype(int) == s] for s in sizes]
    bp1 = ax1.boxplot(sig_data, positions=positions, widths=0.6, patch_artist=True,
                      showfliers=False, medianprops=dict(color='white'))
    for patch in bp1['boxes']:
        patch.set_facecolor('#1976d2')
        patch.set_alpha(0.7)
    ax1.set_ylabel("mu (signal only)")
    ax1.set_title("Signal embedding: expect flattening at mid-range sizes")
    ax1.grid(True, alpha=0.3, axis='y')

    # Full measure
    ful_data = [mu_full[targets.astype(int) == s] for s in sizes]
    bp2 = ax2.boxplot(ful_data, positions=positions, widths=0.6, patch_artist=True,
                      showfliers=False, medianprops=dict(color='white'))
    for patch in bp2['boxes']:
        patch.set_facecolor('#d32f2f')
        patch.set_alpha(0.7)
    ax2.set_ylabel("mu (full)")
    ax2.set_title("Full measure: calibration should unfold mid-range")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel("Circuit size")
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "curvature_boxplot.pdf"), dpi=150)
    plt.close(fig)
    log.info("  Saved curvature_boxplot.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Artificial curvature correction
# ═══════════════════════════════════════════════════════════════════════════

def artificial_correction(genome, measures, quad, targets):
    """Replace sensitivity with mu_signal^2 or piecewise transform."""
    log.info("=" * 70)
    log.info("EXPERIMENT 2: ARTIFICIAL CURVATURE CORRECTION")
    log.info("=" * 70)

    si = MEASURE_NAMES.index("sensitivity")
    se = MEASURE_NAMES.index("spectral_entropy")
    inf_i = MEASURE_NAMES.index("influence")

    # Compute signal-only mu
    g_no_sens = zero_feature(genome, si)
    mu_signal = evaluate_genome_values(g_no_sens, measures, quad)
    r_signal, _ = spearmanr(mu_signal, targets)

    # Full baseline
    mu_full = evaluate_genome_values(genome, measures, quad)
    r_full, _ = spearmanr(mu_full, targets)

    log.info("  r_full    = %.4f", r_full)
    log.info("  r_signal  = %.4f  (no sensitivity)", r_signal)

    results = {"r_full": r_full, "r_signal": r_signal, "corrections": {}}

    # --- Correction A: mu_signal + alpha * mu_signal^2 ---
    log.info("")
    log.info("  --- Correction A: mu + alpha * mu^2 ---")

    def _neg_spearman_quad(alpha):
        mu_corrected = mu_signal + alpha * mu_signal ** 2
        r, _ = spearmanr(mu_corrected, targets)
        return -r

    # grid search first, then refine
    best_alpha = 0
    best_r = r_signal
    alphas = np.linspace(-5, 5, 1001)
    for a in alphas:
        mu_c = mu_signal + a * mu_signal ** 2
        r, _ = spearmanr(mu_c, targets)
        if r > best_r:
            best_r = r
            best_alpha = a

    # refine with scipy
    res = minimize_scalar(_neg_spearman_quad,
                          bounds=(best_alpha - 1, best_alpha + 1),
                          method='bounded')
    alpha_opt = res.x
    r_quad = -res.fun

    log.info("  Optimal alpha = %.4f", alpha_opt)
    log.info("  r_corrected   = %.4f  (recovery: %.1f%% of full)",
             r_quad, 100 * r_quad / r_full)

    results["corrections"]["quadratic"] = {
        "alpha": float(alpha_opt),
        "r": float(r_quad),
        "recovery_pct": 100 * r_quad / r_full,
    }

    # --- Correction B: mu_signal + alpha * mu_signal^3 ---
    log.info("")
    log.info("  --- Correction B: mu + alpha * mu^3 ---")

    def _neg_spearman_cubic(alpha):
        mu_corrected = mu_signal + alpha * mu_signal ** 3
        r, _ = spearmanr(mu_corrected, targets)
        return -r

    best_alpha_c = 0
    best_r_c = r_signal
    for a in alphas:
        mu_c = mu_signal + a * mu_signal ** 3
        r, _ = spearmanr(mu_c, targets)
        if r > best_r_c:
            best_r_c = r
            best_alpha_c = a

    res_c = minimize_scalar(_neg_spearman_cubic,
                            bounds=(best_alpha_c - 1, best_alpha_c + 1),
                            method='bounded')
    alpha_c_opt = res_c.x
    r_cubic = -res_c.fun

    log.info("  Optimal alpha = %.4f", alpha_c_opt)
    log.info("  r_corrected   = %.4f  (recovery: %.1f%% of full)",
             r_cubic, 100 * r_cubic / r_full)

    results["corrections"]["cubic"] = {
        "alpha": float(alpha_c_opt),
        "r": float(r_cubic),
        "recovery_pct": 100 * r_cubic / r_full,
    }

    # --- Correction C: optimal monotonic piecewise (5-knot spline) ---
    log.info("")
    log.info("  --- Correction C: optimal 2-param polynomial (a*mu^2 + b*mu^3) ---")

    def _neg_spearman_poly(params):
        a, b = params
        mu_corrected = mu_signal + a * mu_signal ** 2 + b * mu_signal ** 3
        r, _ = spearmanr(mu_corrected, targets)
        return -r

    from scipy.optimize import minimize
    best_poly = None
    best_poly_r = r_signal
    # multi-start
    for a0 in np.linspace(-3, 3, 7):
        for b0 in np.linspace(-3, 3, 7):
            res_p = minimize(_neg_spearman_poly, [a0, b0], method='Nelder-Mead',
                             options={"maxiter": 200, "xatol": 1e-4})
            if -res_p.fun > best_poly_r:
                best_poly_r = -res_p.fun
                best_poly = res_p.x

    if best_poly is not None:
        log.info("  Optimal: a=%.4f, b=%.4f", best_poly[0], best_poly[1])
        log.info("  r_corrected   = %.4f  (recovery: %.1f%% of full)",
                 best_poly_r, 100 * best_poly_r / r_full)
        results["corrections"]["poly2"] = {
            "a": float(best_poly[0]), "b": float(best_poly[1]),
            "r": float(best_poly_r),
            "recovery_pct": 100 * best_poly_r / r_full,
        }

    # --- Correction D: add raw sensitivity^2 directly (not from genome) ---
    log.info("")
    log.info("  --- Correction D: mu_signal + alpha * sensitivity^2 ---")

    sens_vals = measures[:, si]

    def _neg_spearman_sens2(alpha):
        mu_corrected = mu_signal + alpha * sens_vals ** 2
        r, _ = spearmanr(mu_corrected, targets)
        return -r

    best_alpha_s = 0
    best_r_s = r_signal
    for a in np.linspace(-10, 10, 2001):
        mu_c = mu_signal + a * sens_vals ** 2
        r, _ = spearmanr(mu_c, targets)
        if r > best_r_s:
            best_r_s = r
            best_alpha_s = a

    res_s = minimize_scalar(_neg_spearman_sens2,
                            bounds=(best_alpha_s - 1, best_alpha_s + 1),
                            method='bounded')
    alpha_s_opt = res_s.x
    r_sens2 = -res_s.fun

    log.info("  Optimal alpha = %.4f", alpha_s_opt)
    log.info("  r_corrected   = %.4f  (recovery: %.1f%% of full)",
             r_sens2, 100 * r_sens2 / r_full)

    results["corrections"]["raw_sens_squared"] = {
        "alpha": float(alpha_s_opt),
        "r": float(r_sens2),
        "recovery_pct": 100 * r_sens2 / r_full,
    }

    # --- Correction E: mu_signal + alpha * each_feature^2 (one at a time) ---
    log.info("")
    log.info("  --- Correction E: mu_signal + alpha * feature_i^2 (per feature) ---")

    feature_corrections = {}
    for fi, fname in enumerate(MEASURE_NAMES):
        feat_vals = measures[:, fi]

        def _neg_r(alpha, fv=feat_vals):
            mu_c = mu_signal + alpha * fv ** 2
            r, _ = spearmanr(mu_c, targets)
            return -r

        best_a = 0
        best_r_f = r_signal
        for a in np.linspace(-10, 10, 201):
            mu_c = mu_signal + a * feat_vals ** 2
            r, _ = spearmanr(mu_c, targets)
            if r > best_r_f:
                best_r_f = r
                best_a = a

        res_f = minimize_scalar(_neg_r,
                                bounds=(best_a - 2, best_a + 2),
                                method='bounded')
        r_feat = -res_f.fun
        alpha_feat = res_f.x

        feature_corrections[fname] = {
            "alpha": float(alpha_feat),
            "r": float(r_feat),
            "recovery_pct": 100 * r_feat / r_full,
        }
        log.info("    %-22s  alpha=%+7.3f  r=%.4f  (%.1f%% recovery)",
                 fname, alpha_feat, r_feat, 100 * r_feat / r_full)

    results["corrections"]["per_feature_squared"] = feature_corrections

    # Summary
    log.info("")
    log.info("  === Summary ===")
    log.info("  r_full (original)         = %.4f", r_full)
    log.info("  r_signal (no sensitivity) = %.4f", r_signal)
    log.info("  r + alpha*mu^2            = %.4f  (%.1f%% recovery)",
             r_quad, 100 * r_quad / r_full)
    log.info("  r + alpha*mu^3            = %.4f  (%.1f%% recovery)",
             r_cubic, 100 * r_cubic / r_full)
    if best_poly is not None:
        log.info("  r + a*mu^2 + b*mu^3       = %.4f  (%.1f%% recovery)",
                 best_poly_r, 100 * best_poly_r / r_full)
    log.info("  r + alpha*sens^2 (raw)    = %.4f  (%.1f%% recovery)",
             r_sens2, 100 * r_sens2 / r_full)

    # key question: does mu_signal^2 match sensitivity^2?
    log.info("")
    if r_sens2 > r_quad + 0.01:
        log.info("  [FINDING] sensitivity^2 >> mu_signal^2 as correction")
        log.info("  => sensitivity is a SPECIFIC curvature probe, not generic self-correction")
    elif abs(r_sens2 - r_quad) < 0.01:
        log.info("  [FINDING] sensitivity^2 ~ mu_signal^2 as correction")
        log.info("  => calibration is a GENERIC curvature correction mechanism")
    else:
        log.info("  [FINDING] mu_signal^2 > sensitivity^2 as correction")
        log.info("  => self-correction is more effective than sensitivity probe")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Rank density analysis
# ═══════════════════════════════════════════════════════════════════════════

def rank_density(genome, measures, quad, targets):
    """Per-size rank distributions: show overlap collapse in mid-range."""
    log.info("=" * 70)
    log.info("EXPERIMENT 3: RANK DENSITY ANALYSIS")
    log.info("=" * 70)

    si = MEASURE_NAMES.index("sensitivity")

    mu_full = evaluate_genome_values(genome, measures, quad)
    g_no_sens = zero_feature(genome, si)
    mu_no_sens = evaluate_genome_values(g_no_sens, measures, quad)

    rank_full = rankdata(mu_full, method='average')
    rank_nosens = rankdata(mu_no_sens, method='average')
    N = len(targets)

    sizes = np.unique(targets.astype(int))
    results = {}

    log.info("  Per-size rank statistics:")
    log.info("  %4s  %6s  %8s %8s %8s  %8s %8s %8s  %8s",
             "size", "N",
             "ns_p10", "ns_med", "ns_p90",
             "fl_p10", "fl_med", "fl_p90",
             "sep_gain")

    for s in sizes:
        mask = targets.astype(int) == s
        n = int(mask.sum())

        r_ns = rank_nosens[mask] / N  # normalize to [0, 1]
        r_fl = rank_full[mask] / N

        # separation from neighbors: distance to ideal rank
        ideal_rank = float(np.mean(rankdata(targets, method='average')[mask])) / N

        ns_med = float(np.median(r_ns))
        fl_med = float(np.median(r_fl))

        # separation gain = how much closer to ideal the full measure is
        ns_err = abs(ns_med - ideal_rank)
        fl_err = abs(fl_med - ideal_rank)
        sep_gain = ns_err - fl_err  # positive = full is better

        results[int(s)] = {
            "n": n,
            "no_sens": {
                "p10": float(np.percentile(r_ns, 10)),
                "p25": float(np.percentile(r_ns, 25)),
                "median": float(np.median(r_ns)),
                "p75": float(np.percentile(r_ns, 75)),
                "p90": float(np.percentile(r_ns, 90)),
            },
            "full": {
                "p10": float(np.percentile(r_fl, 10)),
                "p25": float(np.percentile(r_fl, 25)),
                "median": float(np.median(r_fl)),
                "p75": float(np.percentile(r_fl, 75)),
                "p90": float(np.percentile(r_fl, 90)),
            },
            "ideal_rank": ideal_rank,
            "separation_gain": sep_gain,
        }

        log.info("  %4d  %6d  %8.3f %8.3f %8.3f  %8.3f %8.3f %8.3f  %+8.4f",
                 s, n,
                 np.percentile(r_ns, 10), np.median(r_ns), np.percentile(r_ns, 90),
                 np.percentile(r_fl, 10), np.median(r_fl), np.percentile(r_fl, 90),
                 sep_gain)

    # identify the flattened zone: sizes where no_sens has high overlap
    log.info("")
    log.info("  --- Flattened zone detection ---")
    for idx in range(len(sizes) - 1):
        s1, s2 = int(sizes[idx]), int(sizes[idx + 1])
        m1 = targets.astype(int) == s1
        m2 = targets.astype(int) == s2

        # IQR overlap
        ns_iqr1 = (np.percentile(rank_nosens[m1], 25), np.percentile(rank_nosens[m1], 75))
        ns_iqr2 = (np.percentile(rank_nosens[m2], 25), np.percentile(rank_nosens[m2], 75))
        fl_iqr1 = (np.percentile(rank_full[m1], 25), np.percentile(rank_full[m1], 75))
        fl_iqr2 = (np.percentile(rank_full[m2], 25), np.percentile(rank_full[m2], 75))

        def _iqr_overlap(iqr_a, iqr_b):
            lo = max(iqr_a[0], iqr_b[0])
            hi = min(iqr_a[1], iqr_b[1])
            if hi <= lo:
                return 0.0
            span = max(iqr_a[1], iqr_b[1]) - min(iqr_a[0], iqr_b[0])
            return (hi - lo) / span if span > 0 else 0.0

        ns_olap = _iqr_overlap(ns_iqr1, ns_iqr2)
        fl_olap = _iqr_overlap(fl_iqr1, fl_iqr2)
        reduction = ns_olap - fl_olap

        marker = " <-- FLATTENED" if ns_olap > 0.3 else ""
        log.info("    %d-%d: no_sens overlap=%.3f  full overlap=%.3f  reduction=%+.3f%s",
                 s1, s2, ns_olap, fl_olap, reduction, marker)

    # Generate rank density figure
    _plot_rank_density(results, sizes, rank_nosens, rank_full, targets, N)

    return results


def _plot_rank_density(results, sizes, rank_nosens, rank_full, targets, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")
    sizes = sorted(results.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel A: no sensitivity (signal only)
    for i, s in enumerate(sizes):
        mask = targets.astype(int) == s
        r_ns = rank_nosens[mask] / N
        # horizontal violin-like: plot percentile ranges
        p10 = np.percentile(r_ns, 10)
        p25 = np.percentile(r_ns, 25)
        p50 = np.median(r_ns)
        p75 = np.percentile(r_ns, 75)
        p90 = np.percentile(r_ns, 90)

        ax1.barh(i, p90 - p10, left=p10, height=0.6, color='#1976d2', alpha=0.2)
        ax1.barh(i, p75 - p25, left=p25, height=0.6, color='#1976d2', alpha=0.5)
        ax1.plot(p50, i, 'w|', markersize=8, markeredgewidth=2)

    ax1.set_yticks(range(len(sizes)))
    ax1.set_yticklabels(sizes)
    ax1.set_ylabel("Circuit size")
    ax1.set_title("Signal only (no sensitivity): expect overlapping ranges at mid-sizes")
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.2, axis='x')

    # Panel B: full measure
    for i, s in enumerate(sizes):
        mask = targets.astype(int) == s
        r_fl = rank_full[mask] / N
        p10 = np.percentile(r_fl, 10)
        p25 = np.percentile(r_fl, 25)
        p50 = np.median(r_fl)
        p75 = np.percentile(r_fl, 75)
        p90 = np.percentile(r_fl, 90)

        ax2.barh(i, p90 - p10, left=p10, height=0.6, color='#d32f2f', alpha=0.2)
        ax2.barh(i, p75 - p25, left=p25, height=0.6, color='#d32f2f', alpha=0.5)
        ax2.plot(p50, i, 'w|', markersize=8, markeredgewidth=2)

    ax2.set_yticks(range(len(sizes)))
    ax2.set_yticklabels(sizes)
    ax2.set_ylabel("Circuit size")
    ax2.set_xlabel("Normalized rank")
    ax2.set_title("Full measure (signal + calibration): calibration should separate overlapping ranges")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.2, axis='x')

    fig.suptitle("Rank density by circuit size: curvature correction visualization", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rank_density.pdf"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved rank_density.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp = sys.argv[1]
    log.info("Loading data...")
    measures, quad, targets = load_data()
    genome = load_best_genome()

    results = load_results()

    if exp in ("curvature_test", "all"):
        result = curvature_test(genome, measures, quad, targets)
        results["curvature_test"] = result
        save_results(results)

    if exp in ("artificial_correct", "all"):
        result = artificial_correction(genome, measures, quad, targets)
        results["artificial_correction"] = result
        save_results(results)

    if exp in ("rank_density", "all"):
        result = rank_density(genome, measures, quad, targets)
        results["rank_density"] = result
        save_results(results)

    log.info("Done.")


if __name__ == "__main__":
    main()
