"""
Dynamic Programming (DP) baseline for the task allocation problem.

Computes the TRUE optimal makespan (small instances only).

State: f[k][mask] = minimum makespan when the tasks in `mask` are
distributed over k machines.

Recurrence:
    f[k][mask] = min over non-empty subsets s of mask of
                 max( f[k-1][mask without s], sum(s) )

Complexity: O(m * 3^n), so it is only practical for n <= ~14.

Interface:
    optimal_makespan, history = dp(times, m)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from problem import generate_problem, SIZES


def dp(times, m, max_n=14):
   
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n > max_n:
        raise ValueError(
            f"DP is only practical for n <= {max_n} (got n={n}); "
            "complexity is O(m * 3^n)."
        )

    N = 1 << n

    ssum = np.zeros(N)
    for mask in range(1, N):
        low = (mask & -mask).bit_length() - 1
        ssum[mask] = ssum[mask & (mask - 1)] + times[low]

    f = ssum.copy()

    for _ in range(2, m + 1):
        g = f.copy()
        for mask in range(1, N):
            best = g[mask]
            s = mask
            while s:
                v = max(f[mask ^ s], ssum[s])
                if v < best:
                    best = v
                s = (s - 1) & mask
            g[mask] = best
        f = g

    return float(f[N - 1])


if __name__ == "__main__":
    n, m = SIZES["small"]
    times, m = generate_problem(n, m)

    start = time.perf_counter()
    opt = dp(times, m)
    elapsed = time.perf_counter() - start

    print(f"[small] n={n}, m={m} -> DP optimal makespan = {opt:.0f}")
    print(f"lower bound = {max(times.sum() / m, times.max()):.1f}")
    print(f"time = {elapsed:.2f} s")