"""
Relativization barrier check (Baker-Gill-Solovay 1975).

A measure that can be computed purely from the truth table (black-box /
input-output behaviour) will relativize — it cannot separate P from NP
in the presence of an oracle.

Proxy test: fit a low-degree polynomial of the raw truth-table bits to
predict mu.  If R^2 is very high, mu is essentially a polynomial of the
truth table and therefore relativizes.  We WANT the fit to be poor
(measure uses structural information beyond the truth table).
"""

import numpy as np


def _poly_features(bits, degree=2):
    """Expand (N, 32) bit matrix to polynomial features up to `degree`."""
    cols = [bits]
    if degree >= 2:
        D = bits.shape[1]
        ii, jj = np.triu_indices(D)
        cols.append(bits[:, ii] * bits[:, jj])
    return np.concatenate(cols, axis=1)


class RelativizationChecker:
    """Precompute polynomial basis + pseudoinverse for fast batch R^2."""

    def __init__(self, knuth_bits, poly_degree=2, n_samples=10000):
        N = knuth_bits.shape[0]
        if n_samples < N:
            rng = np.random.RandomState(0)
            self._idx = rng.choice(N, size=n_samples, replace=False)
            X_raw = knuth_bits[self._idx]
        else:
            self._idx = np.arange(N)
            X_raw = knuth_bits

        X = _poly_features(X_raw.astype(np.float64), degree=poly_degree)
        XtX = X.T @ X
        reg = 1e-6 * np.eye(XtX.shape[0])
        # pseudoinverse: (p, n_samples)
        self._X = X                                         # (n, p)
        self._pinv = np.linalg.solve(XtX + reg, X.T)       # (p, n)

    def check_batch(self, mu_knuth_batch, r2_threshold=0.99):
        """Check multiple candidates at once.

        mu_knuth_batch: (pop, N_full) — subsample columns via self._idx
        Returns: (passes, r2) arrays of shape (pop,)
        """
        Y = mu_knuth_batch[:, self._idx].astype(np.float64)  # (pop, n)
        W = self._pinv @ Y.T                                 # (p, pop)
        Y_hat = (self._X @ W).T                              # (pop, n)
        ss_res = ((Y - Y_hat) ** 2).sum(axis=1)
        y_mean = Y.mean(axis=1, keepdims=True)
        ss_tot = ((Y - y_mean) ** 2).sum(axis=1)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-20)
        passes = r2 < r2_threshold
        return passes, r2


# legacy single-candidate interface (used by unit barrier wrapper)
def check_relativization(mu_knuth, knuth_bits, *,
                          poly_degree=2, r2_threshold=0.99, n_samples=10000):
    checker = RelativizationChecker(knuth_bits, poly_degree, n_samples)
    passes, r2 = checker.check_batch(mu_knuth[None, :], r2_threshold)
    return bool(passes[0]), float(r2[0])
