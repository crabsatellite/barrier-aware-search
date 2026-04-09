"""
Master runner for all JMLR revision experiments.

Executes experiments in priority order. Checkpoint-resumable at every level:
each experiment script skips already-completed (seed, variant) pairs.

Usage:
    python run_all.py phase1       # critical path (30-seed + synthetic)
    python run_all.py phase2       # domain generalization (SVM + C1/C3 ablations)
    python run_all.py phase3       # robustness (lambda + escape)
    python run_all.py phase4       # polish (mitigation)
    python run_all.py all          # everything, sequential
    python run_all.py status       # show progress across all experiments

GPU: Requires CUDA for Blocks 1A/1B, 3A/3B, 5A, 6A, 7C (Boolean domain).
CPU-only: Blocks 2A (SVM), 2C (synthetic benchmark).
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script, args_list):
    """Run an experiment script as subprocess."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + args_list
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n  WARNING: {script} exited with code {result.returncode}")
    return result.returncode


def phase1():
    """Phase 1 (Week 1-2): Foundation — critical path."""
    print("\n" + "#" * 70)
    print("  PHASE 1: Foundation (30-seed cross-optimizer + synthetic benchmark)")
    print("#" * 70)

    # Block 2C: Synthetic (CPU-only, can run while GPU is busy)
    run_script("exp_synthetic.py", ["run"])

    # Block 1A: 30-seed cross-optimizer (GPU)
    run_script("exp_30seed.py", ["crossopt"])

    # Block 1B: 20-seed landscape (GPU)
    run_script("exp_30seed.py", ["landscape"])


def phase2():
    """Phase 2 (Week 2-3): Domain generalization."""
    print("\n" + "#" * 70)
    print("  PHASE 2: Domain generalization (SVM + C1/C3 ablations)")
    print("#" * 70)

    # Block 2A: SVM fairness (CPU-intensive, no GPU needed)
    run_script("exp_svm_fairness.py", ["run"])

    # Block 3A: C1 ablation - Spearman -> MSE (GPU)
    run_script("exp_ablation.py", ["c1"])

    # Block 3B: C3 ablation - binary -> soft (GPU)
    run_script("exp_ablation.py", ["c3"])


def phase3():
    """Phase 3 (Week 3-4): Robustness."""
    print("\n" + "#" * 70)
    print("  PHASE 3: Robustness (lambda sweep + extended escape)")
    print("#" * 70)

    # Block 5A: Lambda sweep (GPU)
    run_script("exp_lambda_escape.py", ["lambda"])

    # Block 6A: Extended escape (GPU)
    run_script("exp_lambda_escape.py", ["escape"])


def phase4():
    """Phase 4 (Week 4): Polish."""
    print("\n" + "#" * 70)
    print("  PHASE 4: Polish (adaptive mitigation)")
    print("#" * 70)

    # Block 7C: Adaptive sep-switching (GPU)
    run_script("exp_mitigation.py", ["run"])


def status():
    """Show progress for all experiments."""
    print("\n" + "=" * 50)
    print("  JMLR Revision Experiment Status")
    print("=" * 50)

    scripts = [
        ("Phase 1 — 30-seed cross-optimizer", "exp_30seed.py"),
        ("Phase 1 — Synthetic benchmark",    "exp_synthetic.py"),
        ("Phase 2 — SVM fairness",           "exp_svm_fairness.py"),
        ("Phase 2 — C1/C3 ablations",        "exp_ablation.py"),
        ("Phase 3 — Lambda + escape",        "exp_lambda_escape.py"),
        ("Phase 4 — Adaptive mitigation",    "exp_mitigation.py"),
    ]

    for label, script in scripts:
        print(f"\n{label}:")
        run_script(script, ["status"])


def main():
    parser = argparse.ArgumentParser(
        description="JMLR revision experiment runner")
    parser.add_argument("mode",
                        choices=["phase1", "phase2", "phase3", "phase4",
                                 "all", "status"])
    args = parser.parse_args()

    if args.mode == "status":
        status()
    elif args.mode == "phase1":
        phase1()
    elif args.mode == "phase2":
        phase2()
    elif args.mode == "phase3":
        phase3()
    elif args.mode == "phase4":
        phase4()
    elif args.mode == "all":
        phase1()
        phase2()
        phase3()
        phase4()
        print("\n" + "#" * 70)
        print("  ALL PHASES COMPLETE")
        print("#" * 70)
        status()


if __name__ == "__main__":
    main()
