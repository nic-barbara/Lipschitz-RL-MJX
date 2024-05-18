import liprl.utils as utils
import jax.numpy as jnp

from pathlib import Path
from liprl.analysis_tools import ExperimentAnalyser
from liprl.networks.utils import estimate_lipschitz_lower

# Define specifics
env_name = "pendulum"
n_obs = 3
sample_delays = range(5)
attack_sizes = jnp.concatenate((jnp.linspace(0.0, 0.3, 11),
                                jnp.linspace(0.06, 0.15, 10))) # Fine-tune plot
attack_sizes = jnp.unique(attack_sizes.round(2))
attack_sizes.sort()
print("Running attacks: ", attack_sizes)

# Params for perturbations
batches = 1024
horizon = 50
epochs = 20

# Get all file names
dirpath = Path(__file__).resolve().parent
fpath = dirpath / str("../../results/" + env_name + "/trained-models")
files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]

# Create a directory to save results
savepath = dirpath / str("../../results/" + env_name + "/perturbations")
if not savepath.exists():
    savepath.mkdir()

for f in files:
    
    # Load the policy and environment
    print("\nStarting {}".format(f))
    e = ExperimentAnalyser("pendulum", full_fpath=f)
    
    # Compute delayed rewards
    print("Computing delayed rewards...")
    e.setup_batch_eval(n_steps=200, num_envs=batches)
    delayed_rewards = e.eval_perturbed_rewards_mjx(perturbations=sample_delays)
    
    # Computing attacked rewards
    print("Computing attacked rewards...")
    e.setup_batch_eval(n_steps=200, num_envs=batches, 
                       attack_horizon=horizon, attack_epochs=epochs)
    attacked_rewards = e.eval_perturbed_rewards_mjx(perturbations=attack_sizes)
    
    # Estimate the Lipschitz bound of the network (without input normalisation)
    print("Estimating Lipschitz bound...")
    e.config["normalize_observations"] = False
    policy = e._load_policy(return_policy=True)
    gamma = estimate_lipschitz_lower(policy, n_obs)
    e.config["normalize_observations"] = True
    
    # Save all the information
    fname_prefix = str(savepath) + ("/" + env_name + "_" + e.config["network"])
    spath = fname_prefix + ("_perturbed_rewards_" + e.config["version"])
    sdata = {
        "config": e.config, 
        "metrics": e.metrics, 
        "sample_delays": sample_delays,
        "delayed_rewards": delayed_rewards,
        "attack_sizes": attack_sizes,
        "attacked_rewards": attacked_rewards,
        "estimated_lipschitz": gamma
    }
    utils.save_params(spath, sdata)
    