"""
Wait for the main auto-chain (run_remaining.py) to finish,
then re-run the failed blocks to fill in missing trials.
"""
import subprocess
import sys
import os
import time
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

STEPS = [
    ("Block 3B (retry): C3 ablation", ["python", "exp_ablation.py", "c3"]),
    ("Block 5A (retry): Lambda sweep", ["python", "exp_lambda_escape.py", "lambda"]),
]


def main():
    os.chdir(SCRIPT_DIR)

    overall_t0 = time.time()
    for i, (label, cmd) in enumerate(STEPS):
        log_path = os.path.join(LOG_DIR, f"cleanup_{i}.log")
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(STEPS)}] {label}")
        print(f"  Log: {log_path}")
        print(f"{'='*70}\n", flush=True)

        t0 = time.time()
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=SCRIPT_DIR,
            )
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  [{status}] {label} — {elapsed/3600:.1f}h", flush=True)

    total = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"  CLEANUP DONE — total {total/3600:.1f}h")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
