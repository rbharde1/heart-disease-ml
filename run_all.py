"""Reproduce every figure and result of the project in one go.

Runs the four analysis scripts in order, each in its own Python process (the same
interpreter that runs this file), and stops at the first failure:

  1. src/eda.py               exploratory analysis, cleaned data and the train/test split
  2. src/train_evaluate.py    tuning, cross-validation, test-set evaluation, model comparison
  3. src/experiments_data.py  preprocessing ablation, leave-one-site-out, missingness experiments
  4. src/experiments_model.py learning/validation curves, importance, thresholds, calibration, errors

Usage (from the project root):  python run_all.py
Outputs go to figures/ and results/ (results/best_params.json caches the tuned
hyperparameters; delete it to re-tune from scratch).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "src/eda.py",
    "src/train_evaluate.py",
    "src/experiments_data.py",
    "src/experiments_model.py",
]


def run_step(script):
    """Run one script from the project root; return its exit code and elapsed seconds."""
    start = time.perf_counter()
    # -u: unbuffered output, so progress appears live even when the log is redirected
    result = subprocess.run([sys.executable, "-u", script], cwd=ROOT)
    return result.returncode, time.perf_counter() - start


def main():
    total_start = time.perf_counter()
    timings = []
    for i, script in enumerate(STEPS, start=1):
        print(f"\n{'=' * 72}\n[{i}/{len(STEPS)}] {script}\n{'=' * 72}", flush=True)
        code, elapsed = run_step(script)
        timings.append((script, elapsed))
        print(f"--> {script} finished in {elapsed:.1f} s (exit code {code})", flush=True)
        if code != 0:
            print(f"\nStopped: {script} failed with exit code {code}.", flush=True)
            sys.exit(code)

    print(f"\n{'=' * 72}\nSummary")
    for script, elapsed in timings:
        print(f"  {script:<28} {elapsed:7.1f} s")
    print(f"  {'total':<28} {time.perf_counter() - total_start:7.1f} s")


if __name__ == "__main__":
    main()
