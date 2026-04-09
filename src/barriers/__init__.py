from .natural_proof import check_natural_proof
from .relativization import check_relativization, RelativizationChecker
from .algebrization import check_algebrization


_NP_SUBSAMPLE_IDX = None
_NP_SUBSAMPLE_N = 10000


def check_all_barriers_batch(mu_knuth_batch, mu_random_batch, candidates,
                             rel_checker, cfg_barriers,
                             n_vars=5, n_measures=10):
    """Batch barrier check for all candidates in one generation.

    Args:
        mu_knuth_batch:  (pop, N_k) measure values on Knuth functions
        mu_random_batch: (pop, N_r) measure values on random functions
        candidates:      (pop, genome_dim) genomes
        rel_checker:     precomputed RelativizationChecker
        cfg_barriers:    barrier config dict
        n_vars:          number of Boolean variables
        n_measures:      number of base measures (D)

    Returns:
        passes:  (pop,) bool array
        details: list of dicts with per-barrier results
    """
    import numpy as np
    global _NP_SUBSAMPLE_IDX
    pop = len(candidates)

    # --- natural proof (vectorised, subsampled median for speed) ---
    np_cfg = cfg_barriers["natural_proof"]
    poly_deg = np_cfg["density_poly_degree"]
    N_k = mu_knuth_batch.shape[1]
    if N_k > _NP_SUBSAMPLE_N:
        if _NP_SUBSAMPLE_IDX is None or len(_NP_SUBSAMPLE_IDX) != _NP_SUBSAMPLE_N:
            rng = np.random.RandomState(42)
            _NP_SUBSAMPLE_IDX = rng.choice(N_k, _NP_SUBSAMPLE_N, replace=False)
        medians = np.median(np.abs(mu_knuth_batch[:, _NP_SUBSAMPLE_IDX]), axis=1)
    else:
        medians = np.median(np.abs(mu_knuth_batch), axis=1)
    above = np.abs(mu_random_batch) >= medians[:, None]            # (pop, R)
    densities = above.astype(np.float64).mean(axis=1)              # (pop,)
    max_density = 1.0 / (n_vars ** poly_deg)
    np_passes = densities <= max_density

    # --- relativization (batch via precomputed pseudoinverse) ---
    rel_cfg = cfg_barriers["relativization"]
    rel_passes, rel_r2 = rel_checker.check_batch(
        mu_knuth_batch, r2_threshold=rel_cfg["r2_threshold"])

    # --- algebrization (per candidate, fast structural check) ---
    alg_cfg = cfg_barriers["algebrization"]
    alg_passes = np.zeros(pop, dtype=bool)
    alg_scores = np.zeros(pop)
    for i in range(pop):
        ap, asc = check_algebrization(
            candidates[i], n_features=n_measures,
            extensible_indices=alg_cfg["extensible_indices"],
            weight_threshold=alg_cfg["weight_threshold"],
        )
        alg_passes[i] = ap
        alg_scores[i] = asc

    all_pass = np_passes & rel_passes & alg_passes

    details = []
    for i in range(pop):
        details.append({
            "natural_proof":  {"passes": bool(np_passes[i]),
                               "density": float(densities[i])},
            "relativization": {"passes": bool(rel_passes[i]),
                               "r2": float(rel_r2[i])},
            "algebrization":  {"passes": bool(alg_passes[i]),
                               "extensible_weight": float(alg_scores[i])},
        })

    return all_pass, details


# single-candidate wrapper (for analysis scripts)
def check_all_barriers(mu_knuth, mu_random, genome, knuth_bits, cfg_barriers,
                       n_vars=5, n_measures=10):
    np_pass, np_density = check_natural_proof(
        mu_random, mu_knuth,
        n_vars=n_vars,
        poly_degree=cfg_barriers["natural_proof"]["density_poly_degree"],
    )
    rel_pass, rel_r2 = check_relativization(
        mu_knuth, knuth_bits,
        poly_degree=cfg_barriers["relativization"]["poly_degree"],
        r2_threshold=cfg_barriers["relativization"]["r2_threshold"],
        n_samples=cfg_barriers["relativization"]["n_samples"],
    )
    alg_pass, alg_score = check_algebrization(
        genome,
        n_features=n_measures,
        extensible_indices=cfg_barriers["algebrization"]["extensible_indices"],
        weight_threshold=cfg_barriers["algebrization"]["weight_threshold"],
    )

    details = {
        "natural_proof":  {"passes": bool(np_pass),  "density": float(np_density)},
        "relativization": {"passes": bool(rel_pass), "r2": float(rel_r2)},
        "algebrization":  {"passes": bool(alg_pass), "extensible_weight": float(alg_score)},
    }
    return bool(np_pass and rel_pass and alg_pass), details
