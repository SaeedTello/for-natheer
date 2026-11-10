"""
Genetic Algorithm (GA) for the task allocation problem (minimize Makespan).

Solution representation:
    Integer vector of length n, where assign[i] = j means task i runs on machine j.

Fitness: Makespan = load of the busiest machine (lower is better).

Unified interface:
    best_assign, best_makespan, history = ga(times, m, seed=0, ...)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from problem import generate_problem, makespan, print_schedule, SIZES


def _evaluate(pop, times, m):
    loads = np.zeros((pop.shape[0], m))
    for j in range(m):
        loads[:, j] = ((pop == j) * times).sum(axis=1)
    return loads.max(axis=1)


def _tournament_select(pop, fits, k, rng):
    P = pop.shape[0]
    candidates = rng.integers(0, P, size=(P, k))
    winners = candidates[np.arange(P), fits[candidates].argmin(axis=1)]
    return pop[winners]


def ga(times, m, seed=0, pop_size=100, generations=300, pc=0.9,
       pm=None, tournament=3, elite=2, patience=None, max_evals=None):
    """
    Parameters
    
    times       : array of task processing times (length n)
    m           : number of machines
    seed        : random seed (reproducibility)
    pop_size    : population size
    generations : maximum number of generations
    pc          : crossover probability
    pm          : per-gene mutation probability (default 1/n)
    tournament  : tournament size
    elite       : number of best individuals copied unchanged (elitism)
    patience    : stop if no improvement for this many generations (None = disabled)
    max_evals   : evaluation budget (if set, generations are derived from it
                  so that all algorithms can be compared fairly)

    Returns
 
    best_assign   : best assignment found
    best_makespan : its makespan
    history       : best makespan after each generation (Iteration -> Best Fitness)
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    rng = np.random.default_rng(seed)

    if pm is None:
        pm = 1.0 / n
    if elite >= pop_size:
        raise ValueError("elite must be smaller than pop_size")
    if max_evals is not None:
        generations = max(1, max_evals // pop_size - 1)

    pop = rng.integers(0, m, size=(pop_size, n))
    fits = _evaluate(pop, times, m)

    best_idx = fits.argmin()
    best_assign = pop[best_idx].copy()
    best_fit = float(fits[best_idx])
    history = [best_fit]
    stall = 0

    for _ in range(generations):
        parents1 = _tournament_select(pop, fits, tournament, rng)
        parents2 = _tournament_select(pop, fits, tournament, rng)

        do_cross = rng.random(pop_size) < pc
        gene_mask = rng.random((pop_size, n)) < 0.5
        take_from_p2 = do_cross[:, None] & gene_mask
        children = np.where(take_from_p2, parents2, parents1)

        mutate = rng.random((pop_size, n)) < pm
        random_genes = rng.integers(0, m, size=(pop_size, n))
        children = np.where(mutate, random_genes, children)

        if elite > 0:
            elite_idx = fits.argsort()[:elite]
            children[:elite] = pop[elite_idx]

        pop = children
        fits = _evaluate(pop, times, m)

        idx = fits.argmin()
        if fits[idx] < best_fit:
            best_fit = float(fits[idx])
            best_assign = pop[idx].copy()
            stall = 0
        else:
            stall += 1

        history.append(best_fit)

        if patience is not None and stall >= patience:
            break

    return best_assign, best_fit, history


if __name__ == "__main__":
    for name, (n, m) in SIZES.items():
        times, m = generate_problem(n, m)
        assign, best, hist = ga(times, m, seed=0)
        print(f"[{name}] n={n}, m={m} -> GA makespan = {best:.0f}, "
              f"lower bound = {max(times.sum() / m, times.max()):.1f}, "
              f"generations = {len(hist) - 1}")

    times, m = generate_problem(*SIZES["small"])
    assign, best, hist = ga(times, m, seed=0)
    print()
    print_schedule(assign, times, m)



    