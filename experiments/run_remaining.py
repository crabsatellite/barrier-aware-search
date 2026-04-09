"""
Auto-chain all remaining experiments sequentially.
Each block starts automatically after the previous one completes.
"""
import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

STEPS = [
    ("Block 3B: C3 ablation (soft penalty)", ["python", "exp_ablation.py", "c3"]),
    ("Block 5A: Lambda sweep", ["python", "exp_lambda_escape.py", "lambda"]),
    ("Block 6A: Extended escape", ["python", "exp_lambda_escape.py", "escape"]),
    ("Block 7C: Adaptive mitigation", ["python", "exp_mitigation.py", "run"]),
]


def main():
    os.chdir(SCRIPT_DIR)
    overall_t0 = time.time()

    for i, (label, cmd) in enumerate(STEPS):
        log_path = os.path.join(LOG_DIR, f"autochain_{i}.log")
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

        if proc.returncode != 0:
            print(f"  ERROR: check {log_path}", flush=True)

    total = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE — total {total/3600:.1f}h")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
