import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from problem import generate_problem, makespan, SIZES
from algorithms.ga import ga
from algorithms.pso import pso
from algorithms.aco import aco
from algorithms.dp import dp


def check(name, fn, times, m):
    assign, best, hist = fn(times, m, seed=0)
    n = len(times)
    assign = np.asarray(assign)
    lb = max(times.sum() / m, times.max())

    assert assign.shape == (n,), f"{name}: wrong assignment length"
    assert assign.min() >= 0 and assign.max() < m, f"{name}: machine index out of range"
    recomputed = makespan(assign, times, m)
    assert abs(recomputed - best) < 1e-9, f"{name}: reported {best} != recomputed {recomputed}"
    assert best >= lb - 1e-9, f"{name}: makespan below lower bound"
    assert all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1)), \
        f"{name}: history is not non-increasing"
    print(f"OK  {name}: makespan={best:.0f}, lower bound={lb:.1f}")


for size, (n, m) in SIZES.items():
    times, m = generate_problem(n, m)
    for name, fn in [("GA", ga), ("PSO", pso), ("ACO", aco)]:
        check(f"{size}/{name}", fn, times, m)

# Same seed -> same result
times, m = generate_problem(*SIZES["small"])
a = ga(times, m, seed=3)[1]
b = ga(times, m, seed=3)[1]
assert a == b, "GA is not reproducible"
print("OK  reproducibility (same seed -> same result)")

# DP vs brute force on a tiny instance
t = np.array([5, 8, 3, 7, 6, 2, 9])
mm = 3
brute = min(makespan(np.array(c), t, mm)
            for c in itertools.product(range(mm), repeat=len(t)))
assert abs(dp(t, mm) - brute) < 1e-9, "DP does not match brute force"
print(f"OK  DP matches brute force (optimal = {brute:.0f})")

print("\nAll checks passed.")