"""
Algebrization barrier check (Aaronson-Wigderson 2008).

A measure that extends smoothly to algebraic extensions of Boolean
functions will algebrize — it cannot separate complexity classes
beyond what algebrizing techniques can prove.

Structural proxy: check how much of the measure's total weight comes
from algebraically-extensible base features.  If the measure relies
almost entirely on features that have natural algebraic extensions
(degree, nonlinearity, Fourier-based), it is more likely to algebrize.

Extensible features: shannon_entropy (idx 0), spectral_entropy (idx 1),
algebraic_degree (idx 5), nonlinearity (idx 6), autocorrelation_sum (idx 7).

Non-extensible: lz76 (2), run_length (3), gzip_ratio (4),
sensitivity (8), influence (9) — these are Boolean-domain-specific.
"""

import numpy as np


def check_algebrization(genome, *, n_features=10,
                        extensible_indices=None,
                        weight_threshold=0.8):
    """Check whether measure avoids the algebrization barrier.

    Args:
        genome:              full parameter vector
        n_features:          number of base measures (D)
        extensible_indices:  which base features have algebraic extensions
        weight_threshold:    if extensible fraction > this, measure algebrizes

    Returns:
        passes:   True if measure does NOT algebrize
        ext_frac: fraction of total absolute weight on extensible features
    """
    if extensible_indices is None:
        extensible_indices = [0, 1, 5, 6, 7]

    D = n_features
    Q = D * (D + 1) // 2

    w_linear = genome[:D]
    w_quad = genome[D:D + Q]
    w_thresh_act = genome[D + Q:D + Q + D]

    # linear contribution per feature
    abs_linear = np.abs(w_linear)

    # quadratic: distribute weight to constituent features
    abs_quad_per_feat = np.zeros(D)
    idx = 0
    for i in range(D):
        for j in range(i, D):
            half = np.abs(w_quad[idx]) / 2.0
            abs_quad_per_feat[i] += half
            abs_quad_per_feat[j] += half
            idx += 1

    # threshold activation
    abs_thresh = np.abs(w_thresh_act)

    total_per_feat = abs_linear + abs_quad_per_feat + abs_thresh
    total = total_per_feat.sum()
    if total < 1e-12:
        return True, 0.0

    ext_mask = np.zeros(D, dtype=bool)
    ext_mask[extensible_indices] = True
    ext_weight = total_per_feat[ext_mask].sum() / total

    return ext_weight < weight_threshold, float(ext_weight)
