"""
Package A: Mechanism decomposition — ablation, pairwise interaction, threshold activation.
Package B: Expressiveness — representation family stress test.

Answers:
  A1. Where does sensitivity's 69.8% weight come from? (linear / quadratic / threshold)
  A2. Which feature pairs drive the composite? (interaction heatmap)
  A3. Is the measure doing regime segmentation or smooth scoring? (threshold activation)
  B1. Is nonlinear expressiveness necessary or cosmetic? (family comparison)
"""

import json
import logging
import os
import sys
import time

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
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "mechanism.log")),
    ],
)
log = logging.getLogger(__name__)

MEASURE_NAMES = [
    "shannon_entropy", "spectral_entropy", "lz76_complexity",
    "run_length", "gzip_ratio", "algebraic_degree",
    "nonlinearity", "autocorrelation_sum", "sensitivity", "influence",
]
D = len(MEASURE_NAMES)
Q_DIM = D * (D + 1) // 2  # 55
GENOME_DIM = D + Q_DIM + D + D  # 85

# genome layout: [W(10), Q(55), A(10), T(10)]
W_SLICE = slice(0, D)           # linear weights
Q_SLICE = slice(D, D + Q_DIM)   # quadratic weights
A_SLICE = slice(D + Q_DIM, D + Q_DIM + D)  # threshold activations
T_SLICE = slice(D + Q_DIM + D, GENOME_DIM)  # threshold positions


def load_data():
    data = np.load(os.path.join(PROJECT_DIR, "data", "search_features.npz"))
    return data["knuth_measures"], data["knuth_quad"], data["knuth_targets"]


def load_best_genome():
    with open(os.path.join(PROJECT_DIR, "data", "top_candidates.json")) as f:
        results = json.load(f)
    return np.array(results["candidates"][0]["genome"])


def evaluate_genome(genome, measures, quad, targets):
    """Evaluate a single genome, return Spearman r (scipy, correct tie-handling)."""
    from measures import MeasureEvaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MeasureEvaluator(measures, quad, device)
    mu = ev.evaluate_single(genome)
    r, _ = spearmanr(mu, targets)
    return float(r)


def quad_index_to_pair(idx):
    """Map flat upper-triangle index → (i, j) pair."""
    k = 0
    for i in range(D):
        for j in range(i, D):
            if k == idx:
                return i, j
            k += 1
    raise ValueError(f"Invalid quad index {idx}")


# ═══════════════════════════════════════════════════════════════════════════
# A1: Ablation by removal
# ═══════════════════════════════════════════════════════════════════════════

def ablation_study(genome, measures, quad, targets):
    """Zero out specific parameter groups and re-evaluate."""
    log.info("=" * 70)
    log.info("A1: ABLATION STUDY")
    log.info("=" * 70)

    full_r = evaluate_genome(genome, measures, quad, targets)
    log.info("  Full measure: r = %.4f", full_r)

    results = {"full_r": full_r, "ablations": {}}

    # Per-feature ablation for ALL features (not just sensitivity)
    for fi, fname in enumerate(MEASURE_NAMES):
        # Identify all genome indices touching this feature
        # 1. Linear: index fi
        # 2. Quadratic: all (i,j) pairs where i==fi or j==fi
        # 3. Threshold activation + position: A[fi], T[fi]

        ablated = genome.copy()

        # zero linear
        ablated[fi] = 0.0

        # zero quadratic terms involving this feature
        k = 0
        for i in range(D):
            for j in range(i, D):
                if i == fi or j == fi:
                    ablated[D + k] = 0.0
                k += 1

        # zero threshold
        ablated[D + Q_DIM + fi] = 0.0
        # (threshold position T[fi] doesn't matter when A[fi]=0)

        r_ablated = evaluate_genome(ablated, measures, quad, targets)
        drop = full_r - r_ablated

        results["ablations"][f"remove_{fname}_all"] = {
            "r": r_ablated, "drop": drop, "drop_pct": 100 * drop / full_r,
        }
        log.info("  Remove %-22s → r = %.4f  (drop %.4f = %.1f%%)",
                 fname, r_ablated, drop, 100 * drop / full_r)

    log.info("")

    # Deeper ablation for sensitivity (the paradox feature)
    si = MEASURE_NAMES.index("sensitivity")

    # sensitivity linear only
    g = genome.copy()
    g[si] = 0.0
    r = evaluate_genome(g, measures, quad, targets)
    results["ablations"]["remove_sensitivity_linear_only"] = {
        "r": r, "drop": full_r - r,
    }
    log.info("  Remove sensitivity linear only  → r = %.4f  (drop %.4f)", r, full_r - r)

    # sensitivity quadratic only
    g = genome.copy()
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                g[D + k] = 0.0
            k += 1
    r = evaluate_genome(g, measures, quad, targets)
    results["ablations"]["remove_sensitivity_quadratic_only"] = {
        "r": r, "drop": full_r - r,
    }
    log.info("  Remove sensitivity quadratic only → r = %.4f  (drop %.4f)", r, full_r - r)

    # sensitivity threshold only
    g = genome.copy()
    g[D + Q_DIM + si] = 0.0
    r = evaluate_genome(g, measures, quad, targets)
    results["ablations"]["remove_sensitivity_threshold_only"] = {
        "r": r, "drop": full_r - r,
    }
    log.info("  Remove sensitivity threshold only → r = %.4f  (drop %.4f)", r, full_r - r)

    # remove all EXCEPT sensitivity
    g = np.zeros_like(genome)
    g[si] = genome[si]  # linear
    k = 0
    for i in range(D):
        for j in range(i, D):
            if i == si or j == si:
                g[D + k] = genome[D + k]
            k += 1
    g[D + Q_DIM + si] = genome[D + Q_DIM + si]
    g[D + Q_DIM + D + si] = genome[D + Q_DIM + D + si]
    r = evaluate_genome(g, measures, quad, targets)
    results["ablations"]["sensitivity_only_all_terms"] = {
        "r": r, "note": "only sensitivity-related terms kept",
    }
    log.info("  Keep ONLY sensitivity terms       → r = %.4f", r)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# A2: Pairwise interaction heatmap
# ═══════════════════════════════════════════════════════════════════════════

def pairwise_interaction(genome):
    """Extract quadratic weights as a 10×10 interaction matrix."""
    log.info("=" * 70)
    log.info("A2: PAIRWISE INTERACTION MAP")
    log.info("=" * 70)

    q_weights = genome[Q_SLICE]
    interaction = np.zeros((D, D))
    k = 0
    for i in range(D):
        for j in range(i, D):
            interaction[i, j] = q_weights[k]
            interaction[j, i] = q_weights[k]
            k += 1

    # rank by absolute magnitude
    pairs = []
    k = 0
    for i in range(D):
        for j in range(i, D):
            pairs.append((abs(q_weights[k]), i, j, float(q_weights[k])))
            k += 1
    pairs.sort(reverse=True)

    log.info("  Top 10 interaction terms:")
    for rank, (mag, i, j, val) in enumerate(pairs[:10], 1):
        kind = "self" if i == j else "cross"
        log.info("    %2d. %-22s × %-22s  w = %+.6f  (%s)",
                 rank, MEASURE_NAMES[i], MEASURE_NAMES[j], val, kind)

    # sensitivity-specific interactions (use pre-built matrix)
    si = MEASURE_NAMES.index("sensitivity")
    log.info("  Sensitivity interactions:")
    for j in range(D):
        val = interaction[si, j]
        log.info("    sensitivity × %-20s  w = %+.6f", MEASURE_NAMES[j], val)

    return {
        "interaction_matrix": interaction.tolist(),
        "top_pairs": [
            {"i": MEASURE_NAMES[i], "j": MEASURE_NAMES[j],
             "weight": val, "abs_weight": mag}
            for mag, i, j, val in pairs[:20]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# A3: Threshold activation profile
# ═══════════════════════════════════════════════════════════════════════════

def threshold_activation_profile(genome, measures, targets):
    """For each threshold term, compute activation rate per circuit size."""
    log.info("=" * 70)
    log.info("A3: THRESHOLD ACTIVATION PROFILE")
    log.info("=" * 70)

    a_weights = genome[A_SLICE]
    t_positions = genome[T_SLICE]

    sizes = np.unique(targets.astype(int))
    results = {}

    log.info("  Threshold parameters:")
    for i in range(D):
        log.info("    %-22s  a = %+.6f  t = %+.6f",
                 MEASURE_NAMES[i], a_weights[i], t_positions[i])

    log.info("")
    log.info("  Activation rates by circuit size:")
    header = "  size   N      " + "  ".join(f"{n[:6]:>6s}" for n in MEASURE_NAMES)
    log.info(header)

    for s in sizes:
        mask = targets.astype(int) == s
        m_subset = measures[mask]
        n = mask.sum()

        rates = []
        for i in range(D):
            activated = (m_subset[:, i] > t_positions[i]).mean()
            rates.append(float(activated))

        results[int(s)] = {"n": int(n), "activation_rates": rates}
        row = f"  {s:4d} {n:6d}  " + "  ".join(f"{r:6.3f}" for r in rates)
        log.info(row)

    # detect gating transitions (>20% change between adjacent sizes)
    log.info("")
    log.info("  Gating transitions (>20%% change between adjacent sizes):")
    found_transition = False
    for i in range(D):
        if abs(a_weights[i]) < 1e-6:
            continue
        for si in range(len(sizes) - 1):
            s1, s2 = int(sizes[si]), int(sizes[si + 1])
            r1 = results[s1]["activation_rates"][i]
            r2 = results[s2]["activation_rates"][i]
            delta = r2 - r1
            if abs(delta) > 0.20:
                log.info("    %-22s  size %d→%d: %.3f → %.3f (Δ=%+.3f)",
                         MEASURE_NAMES[i], s1, s2, r1, r2, delta)
                found_transition = True
    if not found_transition:
        log.info("    (none found)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# B1: Representation family stress test
# ═══════════════════════════════════════════════════════════════════════════

def family_stress_test(measures, quad, targets):
    """Re-run short CMA-ES searches with restricted genome families."""
    from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
    from barriers import check_all_barriers_batch, RelativizationChecker

    log.info("=" * 70)
    log.info("B1: REPRESENTATION FAMILY STRESS TEST")
    log.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    knuth_eval = MeasureEvaluator(measures, quad, device)
    target_ranks = precompute_target_ranks(targets, device)

    # load config for barriers
    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        cfg = json.load(f)

    data = np.load(os.path.join(PROJECT_DIR, "data", "search_features.npz"))
    random_m = data["random_measures"]
    random_q = data["random_quad"]
    knuth_bits = data["knuth_bits"]
    random_eval = MeasureEvaluator(random_m, random_q, device)
    rel_checker = RelativizationChecker(
        knuth_bits,
        poly_degree=cfg["barriers"]["relativization"]["poly_degree"],
        n_samples=cfg["barriers"]["relativization"]["n_samples"],
    )

    families = {
        "linear_only": {"w": True, "q": False, "a": False},
        "linear_quad": {"w": True, "q": True,  "a": False},
        "linear_threshold": {"w": True, "q": False, "a": True},
        "quad_only": {"w": False, "q": True,  "a": False},
        "threshold_only": {"w": False, "q": False, "a": True},
        "full": {"w": True, "q": True,  "a": True},
    }

    results = {}
    pop_size = 100
    max_gen = 500
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)

    for family_name, mask in families.items():
        log.info("")
        log.info("  --- Family: %s ---", family_name)
        t0 = time.time()

        # run CMA-ES with masked genome
        from search import CMAES
        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=42)

        best_r = 0.0
        best_r_feasible = 0.0
        convergence = []
        n_feasible_best = 0
        stall_count = 0
        prev_best = 0.0

        for gen in range(max_gen):
            candidates = es.ask()

            # mask out disabled parameter groups
            if not mask["w"]:
                candidates[:, W_SLICE] = 0.0
            if not mask["q"]:
                candidates[:, Q_SLICE] = 0.0
            if not mask["a"]:
                candidates[:, A_SLICE] = 0.0
                candidates[:, T_SLICE] = 0.0

            # evaluate
            mu_knuth = knuth_eval.evaluate_batch(candidates)
            mu_random = random_eval.evaluate_batch(candidates)
            corr = spearman_batch_gpu(mu_knuth, target_ranks)

            barrier_pass, barrier_details = check_all_barriers_batch(
                mu_knuth, mu_random, candidates, rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )
            n_violations = np.zeros(pop_size, dtype=np.int32)
            for i in range(pop_size):
                if not barrier_pass[i]:
                    n_violations[i] = sum(
                        1 for v in barrier_details[i].values() if not v["passes"])

            penalty = cfg["search"]["barrier_penalty"]
            l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
            fitness = -corr + penalty * n_violations + l2_lambda * l2_norm

            es.tell(candidates, fitness)

            if es.should_restart():
                es.restart()

            gen_best = float(corr.max())
            feasible_corrs = corr[barrier_pass.astype(bool)]
            gen_best_feas = float(feasible_corrs.max()) if len(feasible_corrs) > 0 else 0.0

            if gen_best > best_r:
                best_r = gen_best
            if gen_best_feas > best_r_feasible:
                best_r_feasible = gen_best_feas
                n_feasible_best = int(barrier_pass.sum())

            # early stop on stall (no improvement for 100 gens)
            if best_r > prev_best + 1e-5:
                prev_best = best_r
                stall_count = 0
            else:
                stall_count += 1

            if gen % 100 == 0 or gen == max_gen - 1:
                convergence.append({
                    "gen": gen + 1,
                    "best_r": best_r,
                    "best_r_feasible": best_r_feasible,
                    "n_feasible": int(barrier_pass.sum()),
                })
                log.info("    gen %4d | best_r %.4f  feasible_best %.4f | feasible %d/%d | sigma %.2e",
                         gen + 1, best_r, best_r_feasible,
                         int(barrier_pass.sum()), pop_size, es.sigma)

            if stall_count >= 200:
                log.info("    Early stop: no improvement for 200 gens")
                convergence.append({
                    "gen": gen + 1, "best_r": best_r,
                    "best_r_feasible": best_r_feasible,
                    "n_feasible": int(barrier_pass.sum()),
                })
                break

        elapsed = time.time() - t0
        results[family_name] = {
            "best_r": best_r,
            "best_r_feasible": best_r_feasible,
            "n_feasible": n_feasible_best,
            "elapsed_s": elapsed,
            "convergence": convergence,
        }
        log.info("  %s: best_r=%.4f  feasible_best=%.4f  (%.0fs)",
                 family_name, best_r, best_r_feasible, elapsed)

    # summary table
    log.info("")
    log.info("  === Family comparison ===")
    log.info("  %-20s  %8s  %8s", "Family", "Best r", "Feas. r")
    for name in families:
        r = results[name]
        log.info("  %-20s  %8.4f  %8.4f", name, r["best_r"], r["best_r_feasible"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# C: Axis-locking experiment
# ═══════════════════════════════════════════════════════════════════════════

def axis_locking_test(genome, measures, quad, targets):
    """Fix one axis (signal or calibrator), re-search the other.

    Tests:
    1. Lock spectral_entropy params → re-search sensitivity module → recovery?
    2. Lock sensitivity module → re-search structural features → recovery?
    3. Lock spectral_entropy → re-search everything EXCEPT spectral_entropy
    4. Lock sensitivity module → re-search everything EXCEPT sensitivity
    """
    from measures import MeasureEvaluator, precompute_target_ranks, spearman_batch_gpu
    from barriers import check_all_barriers_batch, RelativizationChecker
    from search import CMAES

    log.info("=" * 70)
    log.info("C: AXIS-LOCKING EXPERIMENT")
    log.info("=" * 70)

    full_r = evaluate_genome(genome, measures, quad, targets)
    log.info("  Full measure baseline: r = %.4f", full_r)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    knuth_eval = MeasureEvaluator(measures, quad, device)
    target_ranks = precompute_target_ranks(targets, device)

    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        cfg = json.load(f)

    data_all = np.load(os.path.join(PROJECT_DIR, "data", "search_features.npz"))
    random_eval = MeasureEvaluator(data_all["random_measures"],
                                   data_all["random_quad"], device)
    rel_checker = RelativizationChecker(
        data_all["knuth_bits"],
        poly_degree=cfg["barriers"]["relativization"]["poly_degree"],
        n_samples=cfg["barriers"]["relativization"]["n_samples"],
    )

    si = MEASURE_NAMES.index("sensitivity")
    se = MEASURE_NAMES.index("spectral_entropy")

    def _get_sensitivity_indices():
        """All genome indices for sensitivity: linear, quadratic, threshold."""
        indices = [si]  # linear
        k = 0
        for i in range(D):
            for j in range(i, D):
                if i == si or j == si:
                    indices.append(D + k)
                k += 1
        indices.append(D + Q_DIM + si)   # threshold activation
        indices.append(D + Q_DIM + D + si)  # threshold position
        return indices

    def _get_feature_indices(fi):
        """All genome indices for a given feature."""
        indices = [fi]  # linear
        k = 0
        for i in range(D):
            for j in range(i, D):
                if i == fi or j == fi:
                    indices.append(D + k)
                k += 1
        indices.append(D + Q_DIM + fi)
        indices.append(D + Q_DIM + D + fi)
        return indices

    sens_idx = set(_get_sensitivity_indices())
    spec_idx = set(_get_feature_indices(se))

    # Define locking experiments
    experiments = {
        "lock_spectral_search_sensitivity": {
            "locked": spec_idx,
            "desc": "Fix spectral_entropy, re-search sensitivity module only",
            "free_mask": sens_idx - spec_idx,  # sensitivity indices not in spectral
        },
        "lock_sensitivity_search_spectral": {
            "locked": sens_idx,
            "desc": "Fix sensitivity module, re-search spectral_entropy only",
            "free_mask": spec_idx - sens_idx,
        },
        "lock_spectral_search_rest": {
            "locked": spec_idx,
            "desc": "Fix spectral_entropy, re-search everything else",
        },
        "lock_sensitivity_search_rest": {
            "locked": sens_idx,
            "desc": "Fix sensitivity module, re-search everything else",
        },
    }

    pop_size = 100
    max_gen = 500
    l2_lambda = cfg["search"].get("l2_lambda", 0.001)
    penalty = cfg["search"]["barrier_penalty"]
    results = {}

    for exp_name, exp in experiments.items():
        log.info("")
        log.info("  --- %s ---", exp_name)
        log.info("  %s", exp.get("desc", ""))
        t0 = time.time()

        locked_idx = list(exp["locked"])
        if "free_mask" in exp:
            free_idx = list(exp["free_mask"])
        else:
            free_idx = [i for i in range(GENOME_DIM) if i not in exp["locked"]]

        es = CMAES(GENOME_DIM, sigma0=1.0, sigma_max=100.0, sigma_min=1e-6,
                   pop_size=pop_size, seed=42)

        best_r = 0.0
        best_r_feasible = 0.0
        convergence = []
        stall_count = 0
        prev_best = 0.0

        for gen in range(max_gen):
            candidates = es.ask()

            # enforce locking: overwrite locked positions with original genome values
            for idx in locked_idx:
                candidates[:, idx] = genome[idx]
            # zero out positions that are neither locked nor free
            for idx in range(GENOME_DIM):
                if idx not in locked_idx and idx not in free_idx:
                    candidates[:, idx] = 0.0

            mu_knuth = knuth_eval.evaluate_batch(candidates)
            mu_random = random_eval.evaluate_batch(candidates)
            corr = spearman_batch_gpu(mu_knuth, target_ranks)

            barrier_pass, barrier_details = check_all_barriers_batch(
                mu_knuth, mu_random, candidates, rel_checker,
                cfg["barriers"], n_vars=cfg["data"]["n_vars"], n_measures=D,
            )
            n_violations = np.zeros(pop_size, dtype=np.int32)
            for i in range(pop_size):
                if not barrier_pass[i]:
                    n_violations[i] = sum(
                        1 for v in barrier_details[i].values() if not v["passes"])

            l2_norm = np.sqrt((candidates ** 2).sum(axis=1))
            fitness = -corr + penalty * n_violations + l2_lambda * l2_norm

            es.tell(candidates, fitness)
            if es.should_restart():
                es.restart()

            gen_best = float(corr.max())
            feasible_corrs = corr[barrier_pass.astype(bool)]
            gen_best_feas = float(feasible_corrs.max()) if len(feasible_corrs) > 0 else 0.0
            if gen_best > best_r:
                best_r = gen_best
            if gen_best_feas > best_r_feasible:
                best_r_feasible = gen_best_feas

            if best_r > prev_best + 1e-5:
                prev_best = best_r
                stall_count = 0
            else:
                stall_count += 1

            if gen % 100 == 0 or gen == max_gen - 1:
                convergence.append({
                    "gen": gen + 1, "best_r": best_r,
                    "best_r_feasible": best_r_feasible,
                    "n_feasible": int(barrier_pass.sum()),
                })
                log.info("    gen %4d | best_r %.4f  feasible %.4f | feasible %d/%d | sigma %.2e",
                         gen + 1, best_r, best_r_feasible,
                         int(barrier_pass.sum()), pop_size, es.sigma)

            if stall_count >= 200:
                log.info("    Early stop at gen %d", gen + 1)
                convergence.append({
                    "gen": gen + 1, "best_r": best_r,
                    "best_r_feasible": best_r_feasible,
                    "n_feasible": int(barrier_pass.sum()),
                })
                break

        elapsed = time.time() - t0
        recovery = best_r_feasible / full_r if full_r > 0 else 0
        results[exp_name] = {
            "best_r": best_r,
            "best_r_feasible": best_r_feasible,
            "recovery_pct": 100 * recovery,
            "n_locked": len(locked_idx),
            "n_free": len(free_idx),
            "elapsed_s": elapsed,
            "convergence": convergence,
        }
        log.info("  Result: best_r=%.4f  feasible=%.4f  recovery=%.1f%%  (%.0fs)",
                 best_r, best_r_feasible, 100 * recovery, elapsed)

    # summary
    log.info("")
    log.info("  === Axis-locking summary (full r = %.4f) ===", full_r)
    log.info("  %-40s  %8s  %8s  %8s", "Experiment", "Feas. r", "Recovery", "Free dims")
    for name, r in results.items():
        log.info("  %-40s  %8.4f  %7.1f%%  %8d",
                 name, r["best_r_feasible"], r["recovery_pct"], r["n_free"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def generate_figures(ablation, interaction, threshold, family, genome, measures, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Figure 1: Ablation waterfall ---
    ablations = ablation["ablations"]
    # per-feature removal only
    removals = {k: v for k, v in ablations.items() if k.startswith("remove_") and k.endswith("_all")}
    names = [k.replace("remove_", "").replace("_all", "") for k in removals]
    drops = [removals[k]["drop"] for k in removals]
    order = np.argsort(drops)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d32f2f' if d > 0.01 else '#1976d2' if d > 0.001 else '#757575'
              for d in [drops[i] for i in order]]
    ax.barh(range(len(names)), [drops[i] for i in order], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("Drop in Spearman r when removed")
    ax.set_title(f"Feature ablation (full r = {ablation['full_r']:.4f})")
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_waterfall.pdf"), dpi=150)
    plt.close(fig)
    log.info("Saved ablation_waterfall.pdf")

    # --- Figure 2: Interaction heatmap ---
    matrix = np.array(interaction["interaction_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(D))
    ax.set_xticklabels(MEASURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(D))
    ax.set_yticklabels(MEASURE_NAMES, fontsize=8)
    ax.set_title("Quadratic interaction weights")
    plt.colorbar(im, ax=ax, label="Weight", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "interaction_heatmap.pdf"), dpi=150)
    plt.close(fig)
    log.info("Saved interaction_heatmap.pdf")

    # --- Figure 3: Threshold activation by circuit size ---
    sizes = sorted(int(s) for s in threshold.keys())
    # only features with non-trivial threshold weights
    a_weights = genome[A_SLICE]
    active_features = [i for i in range(D) if abs(a_weights[i]) > 1e-4]

    if active_features:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in active_features:
            rates = [threshold[str(s)]["activation_rates"][i] for s in sizes]
            counts = [threshold[str(s)]["n"] for s in sizes]
            ax.plot(sizes, rates, 'o-', label=f"{MEASURE_NAMES[i]} (a={a_weights[i]:+.3f})",
                    markersize=4)
        ax.set_xlabel("Circuit size")
        ax.set_ylabel("Activation rate (fraction > threshold)")
        ax.set_title("Threshold gating by circuit complexity")
        ax.legend(fontsize=7, loc="best")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)

        # add sample counts as secondary info
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(sizes)
        ax2.set_xticklabels([f"n={threshold[str(s)]['n']}" for s in sizes],
                            fontsize=6, rotation=45)
        ax2.set_xlabel("Sample count", fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "threshold_activation.pdf"), dpi=150)
        plt.close(fig)
        log.info("Saved threshold_activation.pdf")

    # --- Figure 4: Family comparison ---
    fam_names = list(family.keys())
    fam_r = [family[f]["best_r_feasible"] for f in fam_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_fam = ['#4caf50' if f == 'full' else '#2196f3' for f in fam_names]
    bars = ax.bar(range(len(fam_names)), fam_r, color=colors_fam)
    ax.set_xticks(range(len(fam_names)))
    ax.set_xticklabels([f.replace("_", "\n") for f in fam_names], fontsize=9)
    ax.set_ylabel("Best feasible Spearman r")
    ax.set_title("Representation family expressiveness")
    for bar, r in zip(bars, fam_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{r:.3f}", ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "family_comparison.pdf"), dpi=150)
    plt.close(fig)
    log.info("Saved family_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args()

    t_start = time.time()
    measures, quad, targets = load_data()
    genome = load_best_genome()

    log.info("Loaded: %d functions, %d features, genome dim %d",
             measures.shape[0], measures.shape[1], len(genome))

    ablation = interaction = threshold = family = axis_lock = None

    if args.step in ("A", "all"):
        ablation = ablation_study(genome, measures, quad, targets)
        interaction = pairwise_interaction(genome)
        threshold = threshold_activation_profile(genome, measures, targets)

    if args.step in ("B", "all"):
        family = family_stress_test(measures, quad, targets)

    if args.step in ("C", "all"):
        axis_lock = axis_locking_test(genome, measures, quad, targets)

    # load existing results for partial runs
    result_path = os.path.join(PROJECT_DIR, "data", "mechanism_results.json")
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)

    out = {
        "ablation": ablation or existing.get("ablation"),
        "interaction": interaction or existing.get("interaction"),
        "threshold_activation": threshold or existing.get("threshold_activation"),
        "family_stress_test": family or existing.get("family_stress_test"),
        "axis_locking": axis_lock or existing.get("axis_locking"),
    }

    # Figures (only if both A and B data available)
    if out["ablation"] and out["family_stress_test"]:
        log.info("")
        log.info("Generating figures...")
        try:
            generate_figures(out["ablation"], out["interaction"],
                             out["threshold_activation"], out["family_stress_test"],
                             genome, measures, targets)
        except Exception as e:
            log.warning("Figure generation failed: %s", e)

    with open(result_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("All results saved to %s", result_path)

    total = time.time() - t_start
    log.info("Total elapsed: %.1f min", total / 60)


if __name__ == "__main__":
    main()
