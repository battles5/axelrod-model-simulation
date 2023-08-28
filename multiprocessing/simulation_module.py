import numpy as np

grid_size = 50  # size of the grid
F = 3  # number of features

def simulation_step(cg, q):
    x, y = np.random.randint(grid_size, size=2)
    dx, dy = np.random.choice([-1, 0, 1], size=2)
    nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
    if any(cg[x, y, :] == cg[nx, ny, :]):
        differing_features = np.where(cg[x, y, :] != cg[nx, ny, :])[0]
        if len(differing_features) > 0:
            f = np.random.choice(differing_features)
            cg[x, y, f] = cg[nx, ny, f]
    return cg

def calculate_Smax_norm(cg, q):
    L, _, F = cg.shape
    culture_int = np.sum(cg * (q ** np.arange(F)), axis=2)
    unique_vals, counts = np.unique(culture_int, return_counts=True)
    max_count = np.max(counts)
    Smax_norm = max_count / (L ** 2)
    return Smax_norm

def single_q_simulation(q):
    culture_grid = np.random.randint(q, size=(grid_size, grid_size, F))
    smax_values = []
    previous_culture_grid = np.copy(culture_grid)
    consecutive_cycles = 0
    max_consecutive_cycles = 40000
    for step in range(60000000):
        culture_grid = simulation_step(culture_grid, q)
        smax = calculate_Smax_norm(culture_grid, q)
        smax_values.append(smax)
        if np.array_equal(culture_grid, previous_culture_grid):
            consecutive_cycles += 1
        else:
            consecutive_cycles = 0
        if consecutive_cycles >= max_consecutive_cycles:
            break
        previous_culture_grid = np.copy(culture_grid)
    return np.mean(smax_values)