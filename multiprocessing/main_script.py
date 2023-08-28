import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, freeze_support
from simulation_module import single_q_simulation

if __name__ == '__main__':
    freeze_support()

    num_cores = 4
    q_values = range(10, 401, 10)

    with Pool(num_cores) as p:
        results = list(tqdm(p.imap(single_q_simulation, q_values), total=len(q_values)))

    plt.figure(figsize=(10, 6))
    plt.plot(q_values, results, '-o')
    plt.xlabel('q')
    plt.ylabel('$S_{max}/L$')
    plt.title('$S_{max}/L$ vs q')
    plt.grid(True)
    plt.show()