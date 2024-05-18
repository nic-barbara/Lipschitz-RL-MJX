import os
os.environ['MUJOCO_GL']="egl"

import liprl.utils as utils
import numpy as np
import utils as pendulum

from liprl.analysis_tools import ExperimentAnalyser
from pathlib import Path

# Plot formatting
do_title = False
fsize = 20
utils.startup_plotting(font_size=fsize)

def get_argmax_reward(data):
    r = np.array([d["final_reward"] for d in data])
    return data[r.argmax()]


def get_savepath_prefix(savepath, env_name, network_name):
    name = savepath / (env_name + "_" + network_name)
    return str(name)


# Get all filenames for the tuned models
env_name = "pendulum"
dirpath = Path(__file__).resolve().parent
fpath = dirpath / str("../../results/" + env_name + "/trained-models")
files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]

# Place to save results
savepath = dirpath / f"../../results/paper-plots/best-models/"
if not savepath.exists():
    savepath.mkdir()

# Keep track of final rewards
data = []
for f in files:
    d = utils.load_params(f)
    r = d[3]["rewards"]
    data.append({
        "config": d[2],
        "final_reward": r[-1],
        "fname": f
    })

# Get the unique networks
network_types = ["mlp", "lbdn"]
lipschitz_bounds = [4.0]

# For each network, analyse policy with highest reward
for network in network_types:      
    for g in lipschitz_bounds:
        
        # Get all data for this network type
        if network == "mlp":
            name = network.upper()
            network_data = [d for d in data if d["config"]["network"] == network]
        else:
            name = network.upper() + f" ($\gamma = {g}$)"
            network_data = [d for d in data if (d["config"]["network"] == network) 
                            and (d["config"]["gamma"] == g)]
        best_policy_data = get_argmax_reward(network_data)

        # Analysis options
        fpath = best_policy_data["fname"]
        sname = get_savepath_prefix(savepath, env_name, network)
        delay = 2
        attack = 0.11
        max_clim = 13.0
        
        # Load trained policy
        e = ExperimentAnalyser(env_name, full_fpath=fpath)
        
        # Make a video
        e.config["seed"] = 42
        e.eval_mjx(n_steps=200, fname_prefix=sname)
    
        # Contour map of policy
        fname = sname + "_contours_" + e.config["version"] + ".pdf"
        pendulum.plot_pendulum_contours(e.policy, fname)
        
        # Contour map of local Lipschitz bounds
        fname = sname + "_contours_lip_" + e.config["version"] + ".pdf"
        pendulum.plot_pendulum_lipschitz_contours(e.policy, fname, max_clim=max_clim)
        
        # Trajectories in time domain (no delay)
        fname = sname + "_time_trajectories_" + e.config["version"] + ".pdf"
        pendulum.plot_pendulum_timedomain_sims(e, fname)
        
        # Trajectories in time domain (sample delay 2)
        fname = sname + "_time_trajectories_delayed_" + e.config["version"] + "_d2.pdf"
        pendulum.plot_pendulum_timedomain_sims(e, fname, perturbation=delay)
        
        # Trajectories in time domain (attack = 0.011)
        fname = sname + "_time_trajectories_attacked_" + e.config["version"] + "_a0.11.pdf"
        pendulum.plot_pendulum_timedomain_sims(e, fname, perturbation=attack,
                                               attack_horizon=50, attack_epochs=20)
                
        if network == "mlp":
            break
