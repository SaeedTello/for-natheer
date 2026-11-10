import numpy as np

SIZES = {
    "small":  (12, 3),
    "medium": (50, 5),
    "large":  (200, 10),
}


def generate_problem(n_tasks, n_machines, seed=42):
    rng = np.random.default_rng(seed)
    times = rng.integers(1, 101, size=n_tasks)
    return times, n_machines


def loads(assign, times, m):
    return np.bincount(assign, weights=times, minlength=m)


def makespan(assign, times, m):
    return loads(assign, times, m).max()


def print_schedule(assign, times, m):
    print("Task Scheduling Problem")
    print("-----------------------")
    L = loads(assign, times, m)
    for j in range(m):
        tasks = [f"T{i + 1}" for i in range(len(times)) if assign[i] == j]
        print(f"Machine {j + 1}: {tasks}")
        print(f"Total processing time: {int(L[j])}")
    print("-----------------------")
    print("Makespan:", int(L.max()))


if __name__ == "__main__":
    times, m = generate_problem(*SIZES["small"])
    print("Task times:", times)
    rng = np.random.default_rng(0)
    assign = rng.integers(0, m, size=len(times))
    print_schedule(assign, times, m)