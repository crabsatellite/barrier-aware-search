"""
Step 3: Analyse top candidates from the barrier-aware search.

- Decode genomes into human-readable weight tables
- Feature importance (which base measures contribute most)
- Diversity analysis (pairwise cosine similarity)
- Cross-validate on held-out split
- Generate figures
"""

import json
import logging
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "analyze.log")),
    ],
)
log = logging.getLogger(__name__)


def load_config():
    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        return json.load(f)


def decode_genome(genome, measure_names):
    D = len(measure_names)
    Q = D * (D + 1) // 2
    w = genome[:D]
    q = genome[D:D + Q]
    a = genome[D + Q:D + Q + D]
    t = genome[D + Q + D:D + Q + 2 * D]

    linear = {measure_names[i]: float(w[i]) for i in range(D)}

    quad = {}
    idx = 0
    for i in range(D):
        for j in range(i, D):
            if abs(q[idx]) > 1e-6:
                key = f"{measure_names[i]} * {measure_names[j]}"
                quad[key] = float(q[idx])
            idx += 1

    threshold = {}
    for i in range(D):
        if abs(a[i]) > 1e-6:
            threshold[measure_names[i]] = {
                "activation": float(a[i]),
                "position": float(t[i]),
            }

    return {"linear": linear, "quadratic": quad, "threshold": threshold}


def feature_importance(genome, D):
    """Per-feature absolute weight contribution."""
    Q = D * (D + 1) // 2
    w = np.abs(genome[:D])
    q = genome[D:D + Q]
    a = np.abs(genome[D + Q:D + Q + D])

    q_per_feat = np.zeros(D)
    idx = 0
    for i in range(D):
        for j in range(i, D):
            half = abs(q[idx]) / 2.0
            q_per_feat[i] += half
            q_per_feat[j] += half
            idx += 1

    total = w + q_per_feat + a
    if total.sum() > 0:
        total /= total.sum()
    return total


def pairwise_cosine(genomes):
    norms = np.linalg.norm(genomes, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = genomes / norms
    return normed @ normed.T


def cross_validate(candidates_json, data_path):
    """Evaluate top candidates on held-out test split."""
    try:
        import torch
        from measures import MeasureEvaluator, genome_dim, spearman_batch_gpu, precompute_target_ranks
    except ImportError:
        log.warning("torch not available — skipping cross-validation")
        return None

    prepared = np.load(
        os.path.join(PROJECT_DIR,
                     load_config()["data"]["prepared_npz"]))
    features = prepared["features"]
    targets = prepared["targets"]
    test_idx = prepared["test_idx"]

    cfg = load_config()
    lo, hi = cfg["data"]["measure_indices"]
    test_m = features[test_idx, lo:hi]
    test_t = targets[test_idx]

    from measures import build_quadratic
    test_q = build_quadratic(test_m)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MeasureEvaluator(test_m, test_q, device)
    tr = precompute_target_ranks(test_t, device)

    genomes = np.array([c["genome"] for c in candidates_json])
    mu = ev.evaluate_batch(genomes)
    corr = spearman_batch_gpu(mu, tr)
    return corr.tolist()


def main():
    cfg = load_config()
    measure_names = cfg["data"]["measure_names"]
    D = len(measure_names)

    cand_path = os.path.join(PROJECT_DIR, "data", "top_candidates.json")
    if not os.path.exists(cand_path):
        log.error("No top_candidates.json — run search first")
        sys.exit(1)

    with open(cand_path) as f:
        results = json.load(f)

    candidates = results["candidates"]
    log.info("Loaded %d candidates from %d generations (%.1f min)",
             len(candidates), results["n_generations"],
             results["total_time_s"] / 60)

    # feasible only
    feasible = [c for c in candidates if c.get("feasible", False)]
    log.info("Feasible candidates: %d / %d", len(feasible), len(candidates))

    analysis = {
        "n_total": len(candidates),
        "n_feasible": len(feasible),
        "n_generations": results["n_generations"],
        "candidates": [],
    }

    # analyse each feasible candidate
    for rank, c in enumerate(feasible[:20]):
        genome = np.array(c["genome"])
        decoded = decode_genome(genome, measure_names)
        importance = feature_importance(genome, D)

        entry = {
            "rank": rank + 1,
            "spearman_r": c["corr"],
            "generation": c["generation"],
            "barrier_details": c.get("barrier_details", {}),
            "decoded_weights": decoded,
            "feature_importance": {
                measure_names[i]: float(importance[i]) for i in range(D)
            },
        }
        analysis["candidates"].append(entry)

        log.info("  #%d  r=%.4f  gen=%d  top features: %s",
                 rank + 1, c["corr"], c["generation"],
                 ", ".join(
                     f"{measure_names[i]}={importance[i]:.3f}"
                     for i in np.argsort(importance)[::-1][:3]
                 ))

    # diversity
    if len(feasible) >= 2:
        genomes = np.array([c["genome"] for c in feasible[:20]])
        cos_sim = pairwise_cosine(genomes)
        mean_sim = float((cos_sim.sum() - np.trace(cos_sim))
                         / max(len(genomes) * (len(genomes) - 1), 1))
        analysis["diversity"] = {
            "mean_cosine_similarity": mean_sim,
            "n_compared": len(genomes),
        }
        log.info("Diversity: mean cosine similarity = %.4f", mean_sim)

    # cross-validation
    log.info("Cross-validating on held-out test split…")
    cv_corrs = cross_validate(feasible[:20], None)
    if cv_corrs:
        for i, r in enumerate(cv_corrs):
            analysis["candidates"][i]["test_spearman_r"] = r
        log.info("Test correlations: %s",
                 ", ".join(f"{r:.4f}" for r in cv_corrs[:5]))

    # save
    out_path = os.path.join(PROJECT_DIR, "data", "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("Analysis saved to %s", out_path)

    # figures
    try:
        _generate_figures(analysis, feasible)
    except Exception as e:
        log.warning("Figure generation failed: %s", e)


def _generate_figures(analysis, feasible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(PROJECT_DIR, "figures")

    # 1 — feature importance heatmap
    if analysis["candidates"]:
        names = list(analysis["candidates"][0]["feature_importance"].keys())
        matrix = np.array([
            [c["feature_importance"][n] for n in names]
            for c in analysis["candidates"]
        ])
        fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * 0.4)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Candidate rank")
        ax.set_title("Feature importance across top candidates")
        plt.colorbar(im, ax=ax, label="Normalised weight")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "feature_importance.pdf"), dpi=150)
        plt.close(fig)
        log.info("Saved feature_importance.pdf")

    # 2 — convergence curve
    meta_path = os.path.join(PROJECT_DIR, "checkpoints", "search_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        gen_log = meta.get("gen_log", [])
        if gen_log:
            gens = [g["generation"] for g in gen_log]
            best_r = [g["best_corr"] for g in gen_log]
            mean_r = [g["mean_corr"] for g in gen_log]
            feas = [g["barrier_pass_rate"] for g in gen_log]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(gens, best_r, label="best Spearman r", linewidth=1)
            ax1.plot(gens, mean_r, label="mean Spearman r",
                     linewidth=0.5, alpha=0.6)
            ax1.set_ylabel("Spearman r")
            ax1.legend(fontsize=8)
            ax1.set_title("Search convergence")

            ax2.plot(gens, feas, color="green", linewidth=1)
            ax2.set_ylabel("Barrier pass rate")
            ax2.set_xlabel("Generation")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "convergence.pdf"), dpi=150)
            plt.close(fig)
            log.info("Saved convergence.pdf")


if __name__ == "__main__":
    main()
