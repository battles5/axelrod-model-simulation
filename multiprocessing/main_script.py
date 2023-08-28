import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from simulation_module import simulate

L = 100
F = 10
steps = 100000000

def simulation_wrapper(q):
    global L, F, steps
    return simulate(L, F, q, steps)

if __name__ == "__main__":
    q_values = np.linspace(10, 400, 40, dtype=int)

    # Usando solo 6 core
    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(simulation_wrapper, q_values), total=len(q_values)))

    for q, result in zip(q_values, results):
        print(f"q = {q}, Smax/L = {result}")
