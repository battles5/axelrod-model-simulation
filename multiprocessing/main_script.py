import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from simulation_module import simulate
import plotly.graph_objects as go

# Funzione per leggere i valori dal file di configurazione
def read_config(filename="config.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    config = {}
    for line in lines:
        key, value = line.strip().split('=')
        config[key.strip()] = eval(value.strip())
    return config

config = read_config()

L = config["L"]
F = config["F"]
steps = config["steps"]
q_values = np.linspace(config["q_start"], config["q_end"], config["q_step"], dtype=int)

def simulation_wrapper(q):
    return simulate(L, F, q, steps)

import csv

if __name__ == "__main__":
    # Usando solo 6 core
    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(simulation_wrapper, q_values), total=len(q_values)))

    # Creazione del grafico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=q_values,
                             y=results,
                             mode='lines+markers',
                             name='Smax/L vs q',
                             line=dict(color='royalblue', width=2),
                             marker=dict(size=8, color='rgba(255, 182, 193, .9)', line=dict(color='rgba(152, 0, 0, .8)', width=2))
                             ))

    fig.update_layout(
        title={
            'text': 'Smax/L vs q',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(title='q', titlefont_size=16, tickfont_size=14, gridcolor='gray'),
        yaxis=dict(title='Smax/L', titlefont_size=16, tickfont_size=14, gridcolor='gray'),
        plot_bgcolor='rgba(230, 230, 230, 0.8)',  # sfondo chiaro
        width=900,
        height=600,
    )

    fig.show()

    # Esporta i dati in CSV
    with open('graph_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["q", "Smax/L"])
        for q, result in zip(q_values, results):
            writer.writerow([q, result])
