# Lipschitz-Bounded Policy Networks for RL in MJX


This repository contains the code used to produce the results in Section 4.1 of the [paper](): *On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks*. See [here](https://github.com/acfr/Lipschitz-RL-Atari) for the code used to produce the results in Section 4.2.

The code in this repository has been structured so that it is extensible to training and evaluating Lipschitz-bounded policies on any other environment in MJX. Please feel free to install it and play around with your favourite robotic environments, or re-create the figures from our paper. 

## Installation

This code is based on MuJoCo's JAX implementation, [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), and the RL libraries in [Brax](https://github.com/google/brax). There are two ways to install all the dependencies and get started. We recommend the following as the easiest way to get started:
- Use a local install in a virtual environment for all development and results analysis.
- Use the docker image for training models across multiple GPUs on a server/cluster/different workstation.

### System dependencies

All code was tested and developed in Ubuntu 22.04 with CUDA 12.3 and Python 3.10.12. To run any of the code in this repository, you will need a CUDA-compatible NVIDIA GPU.

### Using Python virtual environments

Create a Python virtual environment and activate it:

    python -m venv venv
    source venv/bin/activate

Install all the dependencies and the project package itself from within the project root directory.

    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .

The third line installs the local package `lbnn` itself. The `requirements.txt` file was generated with [`pipreqs`](https://github.com/bndr/pipreqs). **Most importantly,** you will need to set up JAX to find and use your NVIDIA GPU with the following.

    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

All plots use LaTeX font as the default. This requires a local install of LaTeX. Note that the default distribution of LaTeX on Ubuntu is not enough. Install the extra packages with the following:

    sudo apt update && sudo apt upgrade
    sudo apt install texlive-full

### Using Docker 

Build the docker container from the project root directory (this might take a while):

    cd <repo/root/directory/>
    docker build -t lipschitz_rl docker/

This will install all dependencies (including CUDA). Training scripts can then be run with

    ./docker/scripts/train_attack_pendulum.sh

Note that you will first need to specify your local path to the repository in the training script.


## Repository structure

The repository is structured as follows:

- `docker/`: contains the Dockerfile to build the `docker` image, and also contains useful scripts with which to run training experiments from within the `docker` image.

- `liprl/`: contains all the tools used to train models, run experiments, collect data, and analyse results wrapped up in a (locally) pip-installable package.

- `results/`: contains all trained models and results:
    - `results/paper-plots/`: contains all figures included in the paper.
    - `results/pendulum`: contains trained models and adversarial attack/sample delay data used to create the plots in `results/paper-plots/`.

- `scripts/pendulum/`: contains scripts for running and analysing experiments on the pendulum environment. Some explanation on the naming convention below:
    - All scripts with the prefix `train_` re-generate results that are already saved in the `results/pendulum/` folder. Eg: code is provided to tune, train, and attack pendulum swing-up policies.
    - All scripts with the pre-fix `analysis_` re-generate the figures in the `results/paper-plots/` folder.

If you would like to re-generate figures from the paper or re-train any models, please run the scripts in the order they are indexed in the `scripts/pendulum/` directory.

## A note on naming conventions

Section 4.1 of the paper compares unconstrained MLP policies with Lipschitz-bounded policies. Throughout the code, these are referred to as `MLP` and `LBDN` (respectively). The term LBDN stands for Lipschitz-Bounded Deep Network. All LBDN networks are composed from [Sandwich layers](https://github.com/acfr/LBDN).

## Contact

Please contact Nicholas Barbara (nicholas.barbara@sydney.edu.au) with any questions.
