"""
Step 1: Precompute features for barrier-aware evolutionary search.

Loads prepared.npz from canonicalization-leakage, extracts the 10 standardised
measures + circuit-size targets, builds quadratic interaction features,
generates random-function features for the natural-proof barrier check,
and saves everything to data/search_features.npz.

Idempotent: skips if output already exists unless --force is given.
Checkpoint: saves partial random-function computation to resume on crash.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from measures import compute_measures_batch, build_quadratic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "logs", "precompute.log")),
    ],
)
log = logging.getLogger(__name__)


def load_config():
    with open(os.path.join(PROJECT_DIR, "configs", "default.json")) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    args = parser.parse_args()

    cfg = load_config()
    out_path = os.path.join(PROJECT_DIR, "data", "search_features.npz")
    ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "precompute_ckpt.npz")

    if os.path.exists(out_path) and not args.force:
        log.info("Output already exists: %s — skipping (use --force to redo)",
                 out_path)
        return

    # ── load prepared.npz ────────────────────────────────────────────────
    npz_path = os.path.join(PROJECT_DIR, cfg["data"]["prepared_npz"])
    log.info("Loading %s", npz_path)
    data = np.load(npz_path)
    features = data["features"]           # (N, 74) float32
    targets = data["targets"]             # (N,)    float32
    m_mean = data["measure_mean"]         # (10,)
    m_std = data["measure_std"]           # (10,)
    N = features.shape[0]

    lo, hi = cfg["data"]["measure_indices"]
    measures_norm = features[:, lo:hi]    # (N, 10) already standardised
    log.info("Knuth data: %d functions, %d measures, sizes %d–%d",
             N, measures_norm.shape[1], int(targets.min()), int(targets.max()))

    # ── quadratic interactions ───────────────────────────────────────────
    log.info("Building quadratic interactions (upper triangle)…")
    t0 = time.time()
    quad_knuth = build_quadratic(measures_norm)
    log.info("  shape %s, %.1f s", quad_knuth.shape, time.time() - t0)

    # ── random function features ─────────────────────────────────────────
    n_rand = cfg["data"]["n_random_functions"]
    seed = cfg["data"]["random_seed"]
    n_vars = cfg["data"]["n_vars"]

    random_raw = None
    start_idx = 0

    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        random_raw = ckpt["random_raw"]
        start_idx = int(ckpt["done_count"])
        log.info("Resuming random-measure computation from %d / %d", start_idx, n_rand)

    if random_raw is None:
        random_raw = np.zeros((n_rand, 10), dtype=np.float64)

    rng = np.random.RandomState(seed)
    truth_tables = rng.randint(0, 2 ** (1 << n_vars), size=n_rand, dtype=np.int64
                               ).astype(np.uint32)

    chunk = 500
    log.info("Computing measures for %d random functions (chunk=%d)…", n_rand, chunk)
    for i in range(start_idx, n_rand, chunk):
        end = min(i + chunk, n_rand)
        random_raw[i:end] = compute_measures_batch(truth_tables[i:end], n_vars)

        # checkpoint every chunk
        np.savez(ckpt_path, random_raw=random_raw,
                 done_count=np.array(end, dtype=np.int64))

        if (end - start_idx) % 2000 == 0 or end == n_rand:
            log.info("  %d / %d done", end, n_rand)

    # standardise using Knuth statistics
    random_norm = ((random_raw - m_mean) / (m_std + 1e-8)).astype(np.float32)
    quad_random = build_quadratic(random_norm)
    log.info("Random features: measures %s, quad %s",
             random_norm.shape, quad_random.shape)

    # ── truth-table bits for relativisation check ────────────────────────
    knuth_bits = features[:, :32]    # (N, 32) raw truth-table bits

    # ── save ─────────────────────────────────────────────────────────────
    log.info("Saving to %s", out_path)
    np.savez_compressed(
        out_path,
        knuth_measures=measures_norm.astype(np.float32),
        knuth_targets=targets,
        knuth_quad=quad_knuth.astype(np.float32),
        knuth_bits=knuth_bits.astype(np.float32),
        random_measures=random_norm,
        random_quad=quad_random.astype(np.float32),
        measure_mean=m_mean,
        measure_std=m_std,
    )

    # clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        log.info("Checkpoint removed")

    log.info("Done — search_features.npz ready")


if __name__ == "__main__":
    main()
