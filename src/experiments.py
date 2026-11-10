"""
Runs all experiments:
  - 3 algorithms (GA, PSO, ACO) x 3 sizes (small, medium, large) x N runs
  - DP optimal for the small size
  - Same evaluation budget for every algorithm (fair comparison)

Outputs (in results/):
  - results_raw.csv       : one row per run
  - results_summary.csv   : mean / std / best / worst / time per (size, algorithm)
  - histories.npz         : convergence curves of every run
"""

import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from problem import generate_problem, SIZES
from algorithms.ga import ga
from algorithms.pso import pso
from algorithms.aco import aco
from algorithms.dp import dp

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

N_RUNS = 10
PROBLEM_SEED = 42

BUDGETS = {
    "small": 10_000,
    "medium": 30_000,
    "large": 60_000,
}

ALGORITHMS = {
    "GA": lambda times, m, seed, budget: ga(times, m, seed=seed, max_evals=budget),
    "PSO": lambda times, m, seed, budget: pso(times, m, seed=seed, max_evals=budget),
    "ACO": lambda times, m, seed, budget: aco(times, m, seed=seed, max_evals=budget),
}


def run_all(sizes=None, n_runs=N_RUNS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sizes = sizes or list(SIZES.keys())

    rows = []
    histories = {}
    optimal = {}

    for size in sizes:
        n, m = SIZES[size]
        times, m = generate_problem(n, m, seed=PROBLEM_SEED)
        lower_bound = max(times.sum() / m, times.max())
        budget = BUDGETS[size]

        if n <= 14:
            t0 = time.perf_counter()
            optimal[size] = dp(times, m)
            dp_time = time.perf_counter() - t0
            print(f"[{size}] DP optimal = {optimal[size]:.0f} ({dp_time:.2f} s)")
            rows.append({
                "size": size, "n": n, "m": m, "algorithm": "DP", "run": 0,
                "best_makespan": optimal[size], "time_s": dp_time,
                "peak_mem_kb": np.nan, "lower_bound": lower_bound,
            })

        for name, fn in ALGORITHMS.items():
            for run in range(n_runs):
                tracemalloc.start()
                t0 = time.perf_counter()
                _, best, hist = fn(times, m, run, budget)
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                rows.append({
                    "size": size, "n": n, "m": m, "algorithm": name, "run": run,
                    "best_makespan": best, "time_s": elapsed,
                    "peak_mem_kb": peak / 1024, "lower_bound": lower_bound,
                })
                histories[f"{size}|{name}|{run}"] = np.array(hist, dtype=float)

            sub = [r["best_makespan"] for r in rows
                   if r["size"] == size and r["algorithm"] == name]
            print(f"[{size}] {name}: mean={np.mean(sub):.1f}, "
                  f"best={np.min(sub):.0f}, worst={np.max(sub):.0f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, "results_raw.csv"), index=False)

    summary = (
        raw[raw["algorithm"] != "DP"]
        .groupby(["size", "algorithm"])
        .agg(mean=("best_makespan", "mean"),
             std=("best_makespan", "std"),
             best=("best_makespan", "min"),
             worst=("best_makespan", "max"),
             mean_time_s=("time_s", "mean"),
             mean_mem_kb=("peak_mem_kb", "mean"),
             lower_bound=("lower_bound", "first"))
        .reset_index()
    )
    order = {s: i for i, s in enumerate(SIZES.keys())}
    summary["_o"] = summary["size"].map(order)
    summary = summary.sort_values(["_o", "mean"]).drop(columns="_o")
    summary.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)

    np.savez(os.path.join(RESULTS_DIR, "histories.npz"), **histories)

    print("\n=== Summary ===")
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved to: {RESULTS_DIR}")
    return raw, summary  


if __name__ == "__main__":
    run_all()