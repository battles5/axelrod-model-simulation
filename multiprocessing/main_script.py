import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from simulation_module import simulate
import plotly.graph_objects as go
import csv
import os

def read_config(filename="config.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    config = {}
    for line in lines:
        key, value = line.split('=')
        config[key.strip()] = eval(value.strip())
    return config

config = read_config()

L = config["L"]
F = config["F"]
steps = config["steps"]
q_values = np.linspace(config["q_start"], config["q_end"], config["q_step"], dtype=int)
num_cycles = config["num_cycles"]

def simulation_wrapper(q):
    smax_values = [simulate(L, F, q, steps) for _ in range(num_cycles)]
    return np.mean(smax_values)

if __name__ == "__main__":
    # Using only 6 cores
    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(simulation_wrapper, q_values), total=len(q_values)))

    # Creazione del grafico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=q_values,
                             y=results,
                             mode='lines+markers',
                             name='<Smax>/N vs q',
                             line=dict(color='royalblue', width=2),
                             marker=dict(size=8, color='rgba(255, 182, 193, .9)', line=dict(color='rgba(152, 0, 0, .8)', width=2))
                             ))

    fig.update_layout(
        title={
            'text': '<Smax>/N vs q',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(title='q', titlefont_size=16, tickfont_size=14, gridcolor='gray'),
        yaxis=dict(title='<Smax>/N', titlefont_size=16, tickfont_size=14, gridcolor='gray'),
        plot_bgcolor='rgba(230, 230, 230, 0.8)',  # sfondo chiaro
        width=900,
        height=600,
    )

    fig.show()

    # Ensure the 'graph' directory exists
    if not os.path.exists('graph'):
        os.makedirs('graph')

    # Esporta i dati in CSV
    with open('graph/graph_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["q", "<Smax>/N"])
        for q, result in zip(q_values, results):
            writer.writerow([q, result])
