"""
Natural-proof barrier check (Razborov-Rudich 1997).

A complexity measure mu is "natural" if:
  1. Constructive: computable in poly time (all our measures are — always true)
  2. Large: Pr_{random f}[mu(f) >= threshold] > 1/poly(n)

If mu is large, it cannot prove super-polynomial circuit lower bounds
(assuming one-way functions exist).

We WANT mu to NOT be large — i.e. random functions should rarely score high.
"""

import numpy as np


def check_natural_proof(mu_random, mu_knuth, *, n_vars=5, poly_degree=2):
    """Check whether measure avoids the natural-proof barrier.

    Args:
        mu_random:   (R,) measure values on uniformly random functions
        mu_knuth:    (N,) measure values on the Knuth NPN representatives
        n_vars:      number of Boolean variables
        poly_degree: exponent k in the density threshold 1/n^k

    Returns:
        passes:  True if measure is NOT natural (density below threshold)
        density: fraction of random functions with |mu| >= threshold
    """
    threshold = np.median(np.abs(mu_knuth))
    density = float(np.mean(np.abs(mu_random) >= threshold))
    max_density = 1.0 / (n_vars ** poly_degree)
    return density <= max_density, density
