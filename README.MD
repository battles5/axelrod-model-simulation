# Axelrod Model of Cultural Diffusion

## Introduction
The goal of the presented work is rooted in the interdisciplinary domain of complex system physics. Drawing insights from Social Physics literature, this project revolves around the Axelrod model of cultural diffusion, albeit with certain variations. The motivation for choosing this model is well-anchored in a rich framework derived from the intersection of history, physics education, and theoretical constructs pleasantly borrowed from sociology. This codebase serves as an analytical and simulation tool, offering a deep dive into the dynamics of the Axelrod model. The entire analysis encapsulates a theoretical introduction to the model, a meticulous simulation using Python, and the integration of the model with two key variations - noise and media influence.

## The Model
The Axelrod model is designed to represent and investigate the process of cultural diffusion and social influences. Here are shown the representation of agents, the mechanisms of interaction, and the dynamics of cultural traits. Additionally, there will be discussed some variants and key considerations that enhance the understanding of the phase transition phenomenon under the model's dynamics.

In this model, agents are represented on a two-dimensional grid of finite size ![Formula3](https://latex.codecogs.com/svg.image?L), with a total of ![Formula4](https://latex.codecogs.com/svg.image?L%5E2) individual agents identified by integers from 1 to ![Formula4](https://latex.codecogs.com/svg.image?L%5E2). Agents possess ![Formula6](https://latex.codecogs.com/svg.image?F) cultural features, each indexed by an integer ![Formula5](https://latex.codecogs.com/svg.image?f) ranging from 1 to ![Formula6](https://latex.codecogs.com/svg.image?F). An agent's culture is denoted by ![Formula6](https://latex.codecogs.com/svg.image?F) integer variables ![Formula7](https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Csigma_%7Bf%7D(i)%5Cright%5C%7D_%7Bf%5Cin%5Cmathcal%7BF%7D%7D), termed cultural traits. Initially, these traits can take values from 0 to ![Formula12](https://latex.codecogs.com/svg.image?q-1), but they can also follow a Poisson distribution with a mean of ![Formula8](https://latex.codecogs.com/svg.image?q), allowing for positive integer values.

Interactions between agents occur in discrete-time increments, where, at each step, a randomly chosen active agent interacts with one of its four neighbors. The probability of interaction ![Formula9](https://latex.codecogs.com/svg.image?%5C(P_%7Bi,j%7D%5C)) between two agents is based on their cultural similarity and can be zero if they share no cultural features. As a result of their interaction, the active agent changes one of its cultural traits to match that of its neighbor, provided the traits are different.

This model can represent a behavior of opinion alignment, where individuals seek to align their opinions with their counterparts during a discussion. If all traits are already identical or they share no common traits, the interaction has no effect.

The interaction process is divided into three consecutive stages during each iteration:

1. Initially, an active agent ![Formula10](https://latex.codecogs.com/svg.image?i) is chosen, along with a neighbor ![Formula13](https://latex.codecogs.com/svg.image?j) from its immediate surroundings.

2. Subsequently, the likelihood of interaction ![Formula9](https://latex.codecogs.com/svg.image?%5C(P_%7Bi,j%7D%5C)) between the pair ![Formula10](https://latex.codecogs.com/svg.image?i) and ![Formula13](https://latex.codecogs.com/svg.image?j) is determined based on the principle of homophily.

3. In the final stage, if the interaction occurs according to ![Formula9](https://latex.codecogs.com/svg.image?%5C(P_%7Bi,j%7D%5C)), a distinct cultural trait not shared between the two agents is randomly selected. The active agent then adopts the corresponding trait of the target agent. If the agents already possess identical traits or none in common, the interaction remains inconsequential.

The initial stage implies that interactions predominantly occur in close proximity, between an agent and its immediate neighbors. This localized interaction approach distinguishes it from models based on complex interaction networks.

For determining the interaction probability in the second stage, the model emphasizes the principle of homophily: agents with greater cultural similarities are more inclined to interact. This suggests that the likelihood of interaction should correlate with the shared cultural features between the two agents.

## Prerequisites

To run the code, you must have Python installed, preferably version 3.10 or higher.

## Dependencies
- `numpy`: For numerical operations and matrix manipulations.
- `multiprocessing`: To facilitate parallel processing for the simulations.
- `tqdm`: To display progress bars during simulations.
- `plotly.graph_objects`: For visualizing the results of the simulations.

## Code Structure
- `multiprocessing\simulation_module.py`: Contains the main Axelrod Model simulation class and related functions.
- `multiprocessing\config.txt`: Provides the configuration parameters for the simulation.
- `multiprocessing\main_script.py`: The main script that reads configurations, runs simulations, and visualizes the results.

## Instructions for Use
1. Ensure all the dependencies are installed.
2. Set the desired parameters in `config.txt`.
3. Run `main_script.py` to initiate the simulations and generate the results.

## Results and Graphs
Upon successful execution of `main_script.py`, a graph depicting the relationship between parameter ![Formula2](https://latex.codecogs.com/svg.image?q) and normalized the largest domain size
![Formula](https://latex.codecogs.com/svg.image?%5Cfrac%7BS_%7Bmax%7D%7D%7BL%5E2%7D)
 will be displayed. Additionally, the data will be exported to `graph_data.csv` for further analysis.

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

## References

- Axelrod R., *The dissemination of culture*, JCR, 1997. \[1\]
- Lanchier N., *The Axelrod model for the dissemination of culture revisited*, The Annals of Applied Probability, 2012. \[2\]
- Castellano C., Marsili M., Vespignani A., *Nonequilibrium Phase Transition in a Model for Social Influence*, Physical Review Letters, 2000. \[3\]
- Flache A., *Between Monoculture and Cultural Polarization: Agent-based Models of Interplay of Social Influence and Cultural Diversity*, Journal of Archaeological Method and Theory, 2018. \[4\]
