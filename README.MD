# Axelrod Model of Cultural Diffusion

## Introduction
The goal of the presented work is rooted in the interdisciplinary domain of complex system physics. Drawing insights from Social Physics literature, this project revolves around the Axelrod model of cultural diffusion, albeit with certain variations. The motivation for choosing this model is well-anchored in a rich framework derived from the intersection of history, physics education, and theoretical constructs pleasantly borrowed from sociology. This codebase serves as an analytical and simulation tool, offering a deep dive into the dynamics of the Axelrod model. The entire analysis encapsulates a theoretical introduction to the model, a meticulous simulation using Python, and the integration of the model with two key variations - noise and media influence.

## Dependencies
- `numpy`: For numerical operations and matrix manipulations.
- `multiprocessing`: To facilitate parallel processing for the simulations.
- `tqdm`: To display progress bars during simulations.
- `plotly.graph_objects`: For visualizing the results of the simulations.

## Code Structure
- `simulation_module.py`: Contains the main Axelrod Model simulation class and related functions.
- `config.txt`: Provides the configuration parameters for the simulation.
- `main_script.py`: The main script that reads configurations, runs simulations, and visualizes the results.

## Instructions for Use
1. Ensure all the dependencies are installed.
2. Set the desired parameters in `config.txt`.
3. Run `main_script.py` to initiate the simulations and generate the results.

## Results and Graphs
Upon successful execution of `main_script.py`, a graph depicting the relationship between parameter $$ q $$ and normalized the largest domain size
$$ \frac{S_{max}}{L^2} $$ will be displayed. Additionally, the data will be exported to `graph_data.csv` for further analysis.

## Model Variations
This implementation of the Axelrod Model introduces two variations:
- **Noise**: A certain level of randomness or unpredictability introduced into the model.
- **Media Influence**: The impact of external media sources on the cultural diffusion process.

## Analysis using the Axelrod Media Noise Notebook

This section provides an overview of the analysis performed using the `axelrod_media_noise.ipynb` notebook.

### Dependencies
- `numpy`: For numerical operations and array manipulations.
- `matplotlib.pyplot`: For plotting the results and visualizations.
- `tqdm`: To display progress bars during the simulations.

### Model Parameters
The notebook explores the Axelrod Model of Cultural Diffusion using the following parameters:
- Grid size: `{grid_size}`
- Number of cultural features, F: `{F}`
- Number of traits per feature, q: `{q}`
- Noise: `{noise}`
- Media influence: `{media}`

### Simulation Functions
Several functions are defined to simulate the cultural diffusion process:
- Basic simulation step without any variations.
- Simulation step with noise.
- Simulation step with media influence.
- Simulation step incorporating both noise and media influence.

### Results
Four different simulations are executed to understand the model's behavior:
1. Basic simulation without any variations.
2. Simulation incorporating noise.
3. Simulation with media influence.
4. Simulation with both noise and media influence.

Each simulation presents the initial and final state of the culture grid along with the evolution of the normalized size of the largest cultural domain and the number of unique cultures over time.
""".format(grid_size="50", F="3", q="10", noise="0.01", media="0.01")

