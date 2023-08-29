import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from simulation_module import simulate
import plotly.graph_objects as go

L = 50
F = 10
steps = 60000000

def simulation_wrapper(q):
    return simulate(L, F, q, steps)

if __name__ == "__main__":
    q_values = np.linspace(10, 401, 20, dtype=int)

    # Usando solo 6 core
    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(simulation_wrapper, q_values), total=len(q_values)))

    # Creazione del grafico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=q_values, y=results, mode='lines+markers', name='Smax/L vs q'))

    fig.update_layout(title='Smax/L vs q',
                      xaxis=dict(title='q'),
                      yaxis=dict(title='Smax/L'),
                      )

    fig.show()