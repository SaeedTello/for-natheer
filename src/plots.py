"""
Generates plots and tables from the saved experiment results.

Outputs (in results/):
  - convergence_<size>.png : Iteration -> Best Fitness (mean over runs)
  - boxplot_<size>.png     : distribution of final makespan over runs
  - time_comparison.png    : mean execution time per algorithm and size
  - summary_table.md       : summary table (for the report)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import SIZES

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

ALGOS = ["GA", "PSO", "ACO"]
COLORS = {"GA": "tab:blue", "PSO": "tab:orange", "ACO": "tab:green"}


def plot_convergence(histories, raw, size):
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGOS:
        curves = [histories[k] for k in histories.files
                  if k.startswith(f"{size}|{algo}|")]
        if not curves:
            continue
        L = max(len(c) for c in curves)
        padded = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in curves])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(L)
        ax.plot(x, mean, label=algo, color=COLORS[algo])
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=COLORS[algo])

    lb = raw[raw["size"] == size]["lower_bound"].iloc[0]
    ax.axhline(lb, color="red", linestyle="--", label="Lower bound")
    dp_rows = raw[(raw["size"] == size) & (raw["algorithm"] == "DP")]
    if not dp_rows.empty:
        ax.axhline(dp_rows["best_makespan"].iloc[0], color="black",
                   linestyle=":", label="DP optimal")

    n, m = SIZES[size]
    ax.set_title(f"Convergence ({size}: n={n}, m={m})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (Makespan)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"convergence_{size}.png"), dpi=150)
    plt.close(fig)


def plot_boxplot(raw, size):
    sub = raw[(raw["size"] == size) & (raw["algorithm"].isin(ALGOS))]
    data = [sub[sub["algorithm"] == a]["best_makespan"].values for a in ALGOS]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, tick_labels=ALGOS, patch_artist=True)
    for patch, a in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(COLORS[a])
        patch.set_alpha(0.5)

    lb = sub["lower_bound"].iloc[0]
    ax.axhline(lb, color="red", linestyle="--", label="Lower bound")
    n, m = SIZES[size]
    ax.set_title(f"Stability over runs ({size}: n={n}, m={m})")
    ax.set_ylabel("Final Makespan")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"boxplot_{size}.png"), dpi=150)
    plt.close(fig)


def plot_time(summary):
    sizes = list(SIZES.keys())
    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, algo in enumerate(ALGOS):
        vals = []
        for s in sizes:
            row = summary[(summary["size"] == s) & (summary["algorithm"] == algo)]
            vals.append(row["mean_time_s"].iloc[0])
        ax.bar(x + (i - 1) * width, vals, width, label=algo, color=COLORS[algo])

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (seconds, log scale)")
    ax.set_title("Execution time")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "time_comparison.png"), dpi=150)
    plt.close(fig)


def write_table(summary):
    cols = ["size", "algorithm", "mean", "std", "best", "worst",
            "mean_time_s", "mean_mem_kb", "lower_bound"]
    df = summary[cols].round(2)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    with open(os.path.join(RESULTS_DIR, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    raw = pd.read_csv(os.path.join(RESULTS_DIR, "results_raw.csv"))
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "results_summary.csv"))
    histories = np.load(os.path.join(RESULTS_DIR, "histories.npz"))

    for size in SIZES:
        plot_convergence(histories, raw, size)
        plot_boxplot(raw, size)
    plot_time(summary)
    write_table(summary)
    print(f"Plots and table saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()