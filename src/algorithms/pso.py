"""
Particle Swarm Optimization (PSO) for the task allocation problem (minimize Makespan).

Solution representation:
    Each particle position is a real-valued vector x of length n, with every
    component in [0, m). The task-to-machine assignment is obtained by
    decoding: assign[i] = floor(x[i]), so task i runs on machine floor(x[i]).

Fitness: Makespan = load of the busiest machine (lower is better).

Velocity / position update:
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    x = x + v

Unified interface:
    best_assign, best_makespan, history = pso(times, m, seed=0, ...)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from problem import generate_problem, print_schedule, SIZES


def _decode(x, m):
    return np.clip(np.floor(x), 0, m - 1).astype(int)


def _evaluate(positions, times, m):
    assign = _decode(positions, m)
    loads = np.zeros((positions.shape[0], m))
    for j in range(m):
        loads[:, j] = ((assign == j) * times).sum(axis=1)
    return loads.max(axis=1)


def pso(times, m, seed=0, swarm_size=100, iterations=300, w_max=0.9, w_min=0.4,
        c1=2.0, c2=2.0, v_max_ratio=0.5, patience=None, max_evals=None):
    """
    Parameters
    times       : array of task processing times (length n)
    m           : number of machines
    seed        : random seed (reproducibility)
    swarm_size  : number of particles
    iterations  : maximum number of iterations
    w_max       : initial inertia weight
    w_min       : final inertia weight (w decreases linearly from w_max to w_min)
    c1          : cognitive coefficient (attraction to the particle's own best)
    c2          : social coefficient (attraction to the global best)
    v_max_ratio : maximum velocity as a fraction of the search range m
    patience    : stop if no improvement for this many iterations (None = disabled)
    max_evals   : evaluation budget (if set, iterations are derived from it
                  so that all algorithms can be compared fairly)

    Returns
    best_assign   : best assignment found
    best_makespan : its makespan
    history       : best makespan after each iteration (Iteration -> Best Fitness)
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    rng = np.random.default_rng(seed)

    if max_evals is not None:
        iterations = max(1, max_evals // swarm_size - 1)

    v_max = v_max_ratio * m


    upper = m - 1e-9
    x = rng.uniform(0, upper, size=(swarm_size, n))
    v = rng.uniform(-v_max, v_max, size=(swarm_size, n))

    fits = _evaluate(x, times, m)

    pbest = x.copy()
    pbest_fit = fits.copy()
    g_idx = pbest_fit.argmin()
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history = [gbest_fit]
    stall = 0

    for it in range(iterations):
        w = w_max - (w_max - w_min) * (it / max(1, iterations - 1))

        r1 = rng.random((swarm_size, n))
        r2 = rng.random((swarm_size, n))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)

        x = x + v
        x = np.clip(x, 0, upper)

        fits = _evaluate(x, times, m)

        improved = fits < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fits[improved]

        idx = pbest_fit.argmin()
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = float(pbest_fit[idx])
            gbest = pbest[idx].copy()
            stall = 0
        else:
            stall += 1

        history.append(gbest_fit)

        if patience is not None and stall >= patience:
            break

    best_assign = _decode(gbest, m)
    return best_assign, gbest_fit, history


if __name__ == "__main__":
    for name, (n, m) in SIZES.items():
        times, m = generate_problem(n, m)
        assign, best, hist = pso(times, m, seed=0)
        print(f"[{name}] n={n}, m={m} -> PSO makespan = {best:.0f}, "
              f"lower bound = {max(times.sum() / m, times.max()):.1f}, "
              f"iterations = {len(hist) - 1}")

    times, m = generate_problem(*SIZES["small"])
    assign, best, hist = pso(times, m, seed=0)
    print()
    print_schedule(assign, times, m)