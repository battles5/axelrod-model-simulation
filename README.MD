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
Upon successful execution of `main_script.py`, a graph depicting the relationship between parameter ![q](readme_img/q.png) and normalized the largest domain size
![smaxl](readme_img/smax_l2.png) will be displayed. Additionally, the data will be exported to `graph_data.csv` for further analysis.

## Model Variations
This implementation of the Axelrod Model introduces two variations:
- **Noise**: A certain level of randomness or unpredictability introduced into the model.
- **Media Influence**: The impact of external media sources on the cultural diffusion process.