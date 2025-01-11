
<h1 align="center">
  <a href="https://sakana.ai/asal">
    <img width="600" alt="Discovered ALife Simulations" src="https://pub.sakana.ai/asal_blog_assets/cover_video_square-min.png"></a><br>
</h1>


<h1 align="center">
Reproducibility Study of "Learning Perturbations to Explain
Time Series Predictions"
</h1>
<p align="center">
<a href="https://colab.research.google.com/github/SakanaAI/asal/blob/main/asal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

## Abstract
We attempt to reproduce and extend the results presented in Automating the
Search for Artificial Life with Foundation Models, which introduced ASAL, a novel frame-
work for discovering emergent behaviors in artificial life (ALife) simulations using foundation
models. We investigated the core claims of the paper, specifically that (1) ASAL effectively
discovers target phenomena, open-ended simulations, and diverse emergent behaviors across
a variety of ALife substrates, and (2) the use of vision-language foundation models enables
quantitative evaluations of emergent properties that align with human intuition. While
some discrepancies arose in parameter-specific optimizations, our results broadly validate
the primary findings of the original paper. Additionally, we propose new visualizations
for mapping simulation diversity and analyze the sensitivity of ASAL’s metrics to various
substrate configurations. Finally, we evaluate whether ASAL’s open-ended simulations ex-
hibit unintended biases or failure cases and find that the model remains robust across our
tested scenarios. Overall, our work supports and refines the conclusions of the original study
while introducing additional insights into ASAL’s methodology and applications. Code and
experimental details are available at this link.

<div style="display: flex; justify-content: space-between;">
  <img src="https://pub.sakana.ai/asal_blog_assets/teaser.png" alt="Image 1" style="width: 48%;">
  <img src="https://pub.sakana.ai/asal_blog_assets/methods_figure.png" alt="Image 2" style="width: 48%;">
</div>

## Repo Description
This repo contains a minimalistic implementation of ASAL to get you started ASAP.
Everything is implemented in the [Jax framework](https://github.com/jax-ml/jax), making everything end-to-end jittable and very fast.


The important code is here:
- [foundation_models/__init__.py](foundation_models/__init__.py) has the code to create a foundation model.
- [substrates/__init__.py](substrates/__init__.py) has the code to create a substrate.
- [rollout.py](rollout.py) has the code to rollout a simulation efficiently.
- [asal_metrics.py](asal_metrics.py) has the code to compute the metrics from ASAL.

Here is some minimal code to sample some random simulation parameters and run the simulation and evaluate how open-ended it is:
```python
import jax
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics

fm = foundation_models.create_foundation_model('clip')
substrate = substrates.create_substrate('lenia')
rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=substrate.rollout_steps, time_sampling=8, img_size=224, return_state=False) # create the rollout function
rollout_fn = jax.jit(rollout_fn) # jit for speed
# now you can use rollout_fn as you need...
rng = jax.random.PRNGKey(0)
params = substrate.default_params(rng) # sample random parameters
rollout_data = rollout_fn(rng, params)
rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
z = rollout_data['z'] # shape: (8, 512)
oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()
```

We have already implemented the following ALife substrates:
- 'lenia': [Lenia](https://en.wikipedia.org/wiki/Lenia)
- 'boids': [Boids](https://en.wikipedia.org/wiki/Boids)
- 'plife': [Particle Life](https://www.youtube.com/watch?v=scvuli-zcRc)
- 'plife_plus': Particle Life++
  - (Particle Life with changing color dynamics)
- 'plenia': [Particle Lenia](https://google-research.github.io/self-organising-systems/particle-lenia/)
- 'dnca': Discrete Neural Cellular Automata
- 'nca_d1': [Continuous Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- 'gol': [Game of Life/Life-Like Cellular Automata](https://en.wikipedia.org/wiki/Life-like_cellular_automaton)

You can find the code for these substrates at [substrates/](substrates/)

The main files to run the entire ASAL pipeline are the following:
- [main_opt.py](main_opt.py)
  - Run this for supervised target and open-endedness
  - Search algorithm: Sep-CMA-ES (from evosax)
- [main_illuminate.py](main_illuminate.py)
  - Run this for illumination
  - Search algorithm: custom genetic algorithm
- [main_sweep_gol.py](main_sweep_gol.py)
  - Run this for open-endedness in Game of Life substrate (b/c discrete search space)
  - Search algorithm: brute force search

[asal.ipynb](asal.ipynb) goes through everything you need to know.

## Running on Google Colab
<!-- Check out the [Google Colab](here). -->
Check out the Google Colab [here](https://colab.research.google.com/github/SakanaAI/asal/blob/main/asal.ipynb)!

<a href="https://colab.research.google.com/github/SakanaAI/asal/blob/main/asal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Running Locally
### Installation 

To run this project locally, you can start by cloning this repo.
```sh
git clone https://github.com/SakanaAI/asal.git
```
Then, set up the python environment with conda:
```sh
conda create --name asal python=3.10.13
conda activate asal
```

Then, install the necessary python libraries:
```sh
python -m pip install -r requirements.txt
```
However, if you want GPU acceleration (trust me, you do), please [manually install jax](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) according to your system's CUDA version.

### Running ASAL
Check out [asal.ipynb](asal.ipynb) to learn how to run the files and visualize the results.

### Loading Our Dataset of Simulations
You can view our dataset of simulations at
- https://pub.sakana.ai/asal/data/illumination_poster_lenia.png
- https://pub.sakana.ai/asal/data/illumination_poster_boids.png
and find the simulation parameters at
- https://pub.sakana.ai/asal/data/illumination_lenia.npz
- https://pub.sakana.ai/asal/data/illumination_boids.npz
- https://pub.sakana.ai/asal/data/illumination_plife.npz
- https://pub.sakana.ai/asal/data/sweep_gol.npz

Directions on how to load these simulations are shown in [asal.ipynb](asal.ipynb).

## Reproducing Results from the Paper
Everything you need is already in this repo.

If, for some reason, you want to see more code and see what went into the experimentation that led to the creation of ASAL, then check out [this repo](https://github.com/SakanaAI/nca-alife).
  
## Bibtex Citation
To cite our work, you can use the following:
```
@article{kumar2024asal,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Akarsh Kumar and Chris Lu and Louis Kirsch and Yujin Tang and Kenneth O. Stanley and Phillip Isola and David Ha},
  year = {2024},
  url = {https://asal.sakana.ai/}
}
```

