"""
Pipeline orchestrator for barrier-aware search.

Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py --step 2     # resume from step 2
    python run_pipeline.py --step 2 3   # run only steps 2 and 3
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (1, "precompute.py",          "Precompute search features",
     []),
    (2, "search.py",              "CMA-ES barrier-aware search",
     ["--resume"]),
    (3, "analyze_candidates.py",  "Analyse top candidates",
     []),
]


def run_step(num, script, desc, extra_args):
    print(f"\n{'='*60}")
    print(f"  Step {num}: {desc}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + extra_args
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n*** Step {num} FAILED (exit {result.returncode}) "
              f"after {elapsed:.1f}s ***")
        sys.exit(result.returncode)

    print(f"\n  Step {num} done in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Barrier-aware search pipeline")
    parser.add_argument("--step", type=int, nargs="*", default=None,
                        help="Step number(s) to run. "
                             "Single number: run from that step onward. "
                             "Multiple numbers: run only those steps.")
    args = parser.parse_args()

    if args.step is None:
        to_run = STEPS
    elif len(args.step) == 1:
        to_run = [s for s in STEPS if s[0] >= args.step[0]]
    else:
        wanted = set(args.step)
        to_run = [s for s in STEPS if s[0] in wanted]

    if not to_run:
        print("No steps to run.")
        return

    print(f"Running {len(to_run)} step(s): "
          + ", ".join(f"{s[0]}-{s[2]}" for s in to_run))

    total = 0
    for num, script, desc, extra in to_run:
        total += run_step(num, script, desc, extra)

    print(f"\n{'='*60}")
    print(f"  All {len(to_run)} steps complete ({total:.1f}s total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
