# Task Allocation Optimization (Makespan Minimization)

Comparison of three metaheuristics (Genetic Algorithm, Particle Swarm
Optimization, Ant Colony Optimization) against a Dynamic Programming
baseline for assigning tasks to machines so that the makespan (load of the
busiest machine) is minimized.

## Requirements

- Python 3.10+
- Packages: numpy, pandas, matplotlib

```bash
python -m venv .venv
.venv\Scripts\activate       
pip install -r requirements.txt
```

## How to run

Run everything from the project root.

```bash
python src/experiments.py     
python src/plots.py         
python src/sanity_check.py

Each algorithm can also be tested alone:

```bash
python src/algorithms/ga.py
python src/algorithms/pso.py
python src/algorithms/aco.py
python src/algorithms/dp.py
```

## Project structure
```
src/
  problem.py          problem definition (task times, sizes, makespan)
  experiments.py      runs all algorithms and saves results
  plots.py            generates plots and tables
  sanity_check.py     correctness and reproducibility tests
  algorithms/
    ga.py  pso.py  aco.py  dp.py
results/              CSV results and generated plots
requirements.txt
```