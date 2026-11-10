"""
Ant Colony Optimization (ACO) for the task allocation problem (minimize Makespan).

Solution representation:
    assign[i] = j means task i runs on machine j. Each ant builds a full
    assignment by placing tasks one by one.

Pheromone: matrix tau[i][j] = desirability of putting task i on machine j.

Construction rule (probability of choosing machine j for task i):
    P(j) ~ tau[i][j]^alpha * eta^beta
    eta = 1 / (load_j + p_i)   (heuristic: prefer the machine that stays least loaded)

Pheromone update:
    tau = (1 - rho) * tau + deposit, where the best ants deposit Q / makespan.

Fitness: Makespan = load of the busiest machine (lower is better).

Unified interface:
    best_assign, best_makespan, history = aco(times, m, seed=0, ...)
"""


"""
    Parameters
    ----------
    times       : array of task processing times (length n)
    m           : number of machines
    seed        : random seed (reproducibility)
    n_ants      : number of ants per iteration
    iterations  : maximum number of iterations
    alpha       : pheromone importance
    beta        : heuristic importance
    rho         : evaporation rate (0 < rho < 1)
    q           : pheromone deposit constant (default: sum(times) / m, the lower bound)
    tau_min     : minimum pheromone value (avoids probabilities collapsing to zero)
    tau_max     : maximum pheromone value (avoids early stagnation)
    elite_ants  : number of best ants of each iteration that deposit pheromone
    patience    : stop if no improvement for this many iterations (None = disabled)
    max_evals   : evaluation budget (if set, iterations are derived from it
                  so that all algorithms can be compared fairly)

    Returns
    -------
    best_assign   : best assignment found
    best_makespan : its makespan
    history       : best makespan after each iteration (Iteration -> Best Fitness)
    """

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from problem import generate_problem, print_schedule, SIZES


def _evaluate(assigns, times, m):
    loads = np.zeros((assigns.shape[0], m))
    for j in range(m):
        loads[:, j] = ((assigns == j) * times).sum(axis=1)
    return loads.max(axis=1)


def aco(times, m, seed=0, n_ants=50, iterations=300, alpha=1.0, beta=2.0,
        rho=0.1, q=None, tau_min=0.01, tau_max=10.0, elite_ants=3,
        patience=None, max_evals=None):

    times = np.asarray(times, dtype=float)
    n = len(times)
    rng = np.random.default_rng(seed)

    if q is None:
        q = times.sum() / m
    elite_ants = max(1, min(elite_ants, n_ants))
    if max_evals is not None:
        iterations = max(1, max_evals // n_ants)

    tau = np.ones((n, m))

    best_assign = None
    best_fit = float("inf")
    history = []
    stall = 0

    order = np.argsort(-times)

    for _ in range(iterations):
        assigns = np.zeros((n_ants, n), dtype=int)
        loads = np.zeros((n_ants, m))
        ant_idx = np.arange(n_ants)

        for i in order:
            eta = 1.0 / (loads + times[i])                 
            weights = (tau[i][None, :] ** alpha) * (eta ** beta)
            probs = weights / weights.sum(axis=1, keepdims=True)

            cum = probs.cumsum(axis=1)
            r = rng.random((n_ants, 1))
            choice = (r > cum).sum(axis=1)
            choice = np.minimum(choice, m - 1)

            assigns[:, i] = choice
            loads[ant_idx, choice] += times[i]

        fits = loads.max(axis=1)

        idx = fits.argmin()
        if fits[idx] < best_fit:
            best_fit = float(fits[idx])
            best_assign = assigns[idx].copy()
            stall = 0
        else:
            stall += 1

        history.append(best_fit)

        tau *= (1.0 - rho)

        top = fits.argsort()[:elite_ants]
        for a in top:
            deposit = q / fits[a]
            tau[np.arange(n), assigns[a]] += deposit

        tau[np.arange(n), best_assign] += q / best_fit

        tau = np.clip(tau, tau_min, tau_max)

        if patience is not None and stall >= patience:
            break

    return best_assign, best_fit, history


if __name__ == "__main__":
    for name, (n, m) in SIZES.items():
        times, m = generate_problem(n, m)
        assign, best, hist = aco(times, m, seed=0)
        print(f"[{name}] n={n}, m={m} -> ACO makespan = {best:.0f}, "
              f"lower bound = {max(times.sum() / m, times.max()):.1f}, "
              f"iterations = {len(hist)}")

    times, m = generate_problem(*SIZES["small"])
    assign, best, hist = aco(times, m, seed=0)
    print()
    print_schedule(assign, times, m)