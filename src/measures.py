"""
Boolean function measure computation and GPU-batched parameterized evaluation.

Part 1: Raw measure computation (numpy) for arbitrary truth tables
Part 2: GPU-batched parameterized measure evaluation via torch
Part 3: GPU-accelerated Spearman rank correlation
"""

import gzip as gzip_mod
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Part 1 — raw measure computation (numpy, for precomputing random functions)
# ---------------------------------------------------------------------------

def _extract_bits(truth_tables, n=5):
    num_rows = 1 << n
    bits = np.zeros((len(truth_tables), num_rows), dtype=np.float64)
    for j in range(num_rows):
        bits[:, j] = (truth_tables >> j) & 1
    return bits


def _walsh_hadamard(bits, n=5):
    """Batch WHT.  Input: (N, 2^n) in {0,1} → internally {+1,-1} → WHT."""
    f = (1.0 - 2.0 * bits).copy()
    num_rows = 1 << n
    for i in range(n):
        half = 1 << i
        for j in range(0, num_rows, 2 * half):
            for k in range(half):
                u = f[:, j + k].copy()
                v = f[:, j + k + half].copy()
                f[:, j + k] = u + v
                f[:, j + k + half] = u - v
    return f / num_rows  # normalise same as data_prep.py


def _moebius_transform(bits, n=5):
    """ANF coefficients via Möbius over GF(2)."""
    a = bits.astype(np.float32).copy()
    for i in range(n):
        step = 1 << i
        for j in range(1 << n):
            if j & step:
                a[:, j] = np.abs(a[:, j] - a[:, j ^ step])
    return a


def _lz76(s):
    n = len(s)
    if n == 0:
        return 0
    c, i, l = 1, 0, 1
    while i + l < n:
        sub = s[i + 1: i + 1 + l]
        if sub and sub in s[:i + l]:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    norm = n / max(np.log2(n), 1) if n > 1 else 1
    return c / norm


def _longest_run(s):
    if not s:
        return 0
    best = cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


_POPCOUNT_32 = np.array([bin(j).count("1") for j in range(32)])


def compute_measures_batch(truth_tables, n=5):
    """Compute all 10 raw measures for a batch of truth tables.

    Returns (N, 10) float64 matching the column order in prepared.npz:
      [shannon_entropy, spectral_entropy, lz76_complexity, run_length,
       gzip_ratio, algebraic_degree, nonlinearity, autocorrelation_sum,
       sensitivity, influence]
    """
    N = len(truth_tables)
    num_rows = 1 << n
    bits = _extract_bits(truth_tables, n)
    wht = _walsh_hadamard(bits, n)
    anf = _moebius_transform(bits.astype(np.float32), n)

    out = np.zeros((N, 10), dtype=np.float64)

    # 0 — shannon entropy of truth table
    w = bits.sum(axis=1)
    p1 = w / num_rows
    p0 = 1.0 - p1
    ent = np.zeros(N)
    mask = (p0 > 0) & (p1 > 0)
    ent[mask] = -p0[mask] * np.log2(p0[mask]) - p1[mask] * np.log2(p1[mask])
    out[:, 0] = ent

    # 1 — spectral entropy (from WHT)
    wht_sq = wht ** 2
    total = wht_sq.sum(axis=1, keepdims=True)
    total = np.maximum(total, 1e-20)
    prob = wht_sq / total
    lp = np.zeros_like(prob)
    pos = prob > 0
    lp[pos] = np.log2(prob[pos])
    out[:, 1] = -(prob * lp).sum(axis=1)

    # 2 — lz76 complexity  (loop — fast for N ≤ 10 000)
    for i in range(N):
        out[i, 2] = _lz76("".join(str(int(b)) for b in bits[i]))

    # 3 — longest run
    for i in range(N):
        out[i, 3] = _longest_run("".join(str(int(b)) for b in bits[i]))

    # 4 — gzip ratio
    for i in range(N):
        raw = bits[i].astype(np.uint8).tobytes()
        out[i, 4] = len(gzip_mod.compress(raw)) / max(len(raw), 1)

    # 5 — algebraic degree (max Hamming weight of nonzero ANF coefficient)
    pc = _POPCOUNT_32 if num_rows == 32 else np.array(
        [bin(j).count("1") for j in range(num_rows)])
    weighted = anf * pc[None, :]
    out[:, 5] = weighted.max(axis=1)

    # 6 — nonlinearity
    out[:, 6] = (num_rows // 2) - np.abs(wht * num_rows).max(axis=1) / 2

    # 7 — autocorrelation sum (|r_f(s)| for s != 0)
    ac = _walsh_hadamard((wht * num_rows) ** 2 / num_rows, n)
    out[:, 7] = np.abs(ac[:, 1:]).sum(axis=1)

    # 8 — average sensitivity
    sens = np.zeros(N)
    for v in range(n):
        flip = np.arange(num_rows) ^ (1 << v)
        sens += (bits != bits[:, flip]).sum(axis=1)
    out[:, 8] = sens / num_rows

    # 9 — total influence (sum of coordinate influences)
    inf_total = np.zeros(N)
    for v in range(n):
        flip = np.arange(num_rows) ^ (1 << v)
        inf_total += (bits != bits[:, flip]).astype(np.float64).mean(axis=1)
    out[:, 9] = inf_total

    return out


# ---------------------------------------------------------------------------
# Part 2 — GPU-batched parameterized measure evaluation
# ---------------------------------------------------------------------------

def genome_dim(n_features):
    """Total genome length = D + Q + D + D  where Q = D*(D+1)/2."""
    q = n_features * (n_features + 1) // 2
    return n_features + q + n_features + n_features


class MeasureEvaluator:
    """Hold feature tensors on GPU; batch-evaluate candidate genomes."""

    def __init__(self, measures, quad, device):
        self.device = torch.device(device)
        self.F = torch.tensor(measures, dtype=torch.float32, device=self.device)
        self.Q = torch.tensor(quad, dtype=torch.float32, device=self.device)
        self.D = measures.shape[1]
        self.N = measures.shape[0]

    @torch.no_grad()
    def evaluate_batch(self, genomes_np):
        """genomes_np: (pop, genome_dim) ndarray → returns (pop, N) ndarray."""
        D = self.D
        Q_dim = D * (D + 1) // 2
        g = torch.tensor(genomes_np, dtype=torch.float32, device=self.device)
        W = g[:, :D]
        Qw = g[:, D:D + Q_dim]
        A = g[:, D + Q_dim:D + Q_dim + D]
        T = g[:, D + Q_dim + D:D + Q_dim + 2 * D]

        linear = self.F @ W.T                      # (N, pop)
        quadratic = self.Q @ Qw.T                   # (N, pop)

        # threshold term — chunk to cap VRAM
        pop = g.shape[0]
        chunk = max(1, min(pop, int(16e9 / (self.N * D * 4))))
        parts = []
        for s in range(0, pop, chunk):
            e = min(s + chunk, pop)
            diff = self.F.unsqueeze(0) - T[s:e].unsqueeze(1)   # (c, N, D)
            th = torch.einsum("cnd,cd->cn", torch.relu(diff), A[s:e])
            parts.append(th)
        threshold = torch.cat(parts, dim=0)          # (pop, N)

        mu = linear.T + quadratic.T + threshold
        return mu.cpu().numpy()

    @torch.no_grad()
    def evaluate_single(self, genome_np):
        return self.evaluate_batch(genome_np[None, :])[0]


# ---------------------------------------------------------------------------
# Part 3 — GPU-accelerated Spearman rank correlation
# ---------------------------------------------------------------------------

def precompute_target_ranks(targets, device):
    from scipy.stats import rankdata
    # Average ranking for tied values — critical for heavily-tied circuit sizes
    # (293K at size=10, 221K at size=9). Ordinal ranking inflates Spearman r.
    avg_ranks = rankdata(targets, method='average').astype(np.float32)
    return torch.tensor(avg_ranks, device=device)


@torch.no_grad()
def spearman_batch_gpu(mu_np, target_ranks):
    """Batch Spearman: mu_np (pop, N) numpy → (pop,) numpy correlations."""
    dev = target_ranks.device
    mu = torch.tensor(mu_np, dtype=torch.float32, device=dev)
    rx = torch.argsort(torch.argsort(mu, dim=1), dim=1).float()
    rx -= rx.mean(dim=1, keepdim=True)
    ry = target_ranks - target_ranks.mean()
    num = (rx * ry.unsqueeze(0)).sum(dim=1)
    den = rx.norm(dim=1) * ry.norm() + 1e-12
    return (num / den).cpu().numpy()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_quadratic(measures):
    """Upper-triangle pairwise products.  (N, D) → (N, D*(D+1)/2)."""
    D = measures.shape[1]
    ii, jj = np.triu_indices(D)
    return measures[:, ii] * measures[:, jj]
