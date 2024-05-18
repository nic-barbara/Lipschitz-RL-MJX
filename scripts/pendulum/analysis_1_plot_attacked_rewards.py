import liprl.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from liprl.utils import startup_plotting
from pathlib import Path
from utils import order_networks

# Plot formatting
do_title = False
fsize = 20
startup_plotting(font_size=fsize)
perts_to_plot = [0.11] # Which attack values for scatter plot


def get_perturbed_reward_data(data):
    """Extract interesting data to plot for a given network type."""
    
    # Perturbed reward data
    r = np.array([d[f"attacked_rewards"]["rewards"] for d in data])
    
    # Perturbations and Lipschitz bounds
    perturbations = data[0]["attack_sizes"]
    lips = np.array([d["estimated_lipschitz"] for d in data])
    
    out = {
        "attack_sizes": np.array(perturbations),
        "rewards": r.mean(axis=0),
        "stdev": r.std(axis=0),
        "lip_mean": np.array([lips.mean()]),
        "lip_stdev": np.array([lips.std()]),
    }
    return out


# Get all saved data
env_name = "pendulum"
dirpath = Path(__file__).resolve().parent
fpath = dirpath / str("../../results/" + env_name + f"/perturbations")
data = [utils.load_params(f) for f in fpath.iterdir() 
        if f.is_file() and not (f.suffix == ".pdf")]

# Get the unique networks and store data
network_types = utils.lunique([d["config"]["network"] for d in data])
lipschitz_bounds = utils.lunique([d["config"]["gamma"] for d in data
                                  if not d["config"]["network"] == "mlp"])
network_types, lipschitz_bounds = order_networks(network_types, lipschitz_bounds)

results = {}
for network in network_types:
    
    if network == "mlp":
        name_ = "Unconstrained"
        network_data = [d for d in data if d["config"]["network"] == network]
        results[name_] = get_perturbed_reward_data(network_data)
        continue
    
    for g in lipschitz_bounds:
        name_ = "Lipschitz" + " ($\gamma = {:.0f}$)".format(g)
        network_data = [d for d in data if (d["config"]["network"] == network) 
                        and (d["config"]["gamma"] == g)]
        results[name_] = get_perturbed_reward_data(network_data)


# ------------------------------------------------------------------------------
#
#                            Perturbed reward plots
#
# ------------------------------------------------------------------------------

xmax = 0
fig, ax = plt.subplots(figsize=(6.2,5.8))
for network in results:
    x = results[network]["attack_sizes"]
    y = results[network]["rewards"]
    y_err = results[network]["stdev"]
    
    lstyle = "dotted" if network == "Unconstrained" else "solid"
        
    ax.plot()
    ax.plot(x, y, label=network, linestyle=lstyle)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    xmax = np.max((max(x), xmax))
    
if do_title: ax.set_title(f"Effect of adversarial attack ({env_name})")
ax.set_xlabel("Attack size $\epsilon$")
ax.set_ylabel("Reward")
ax.set_xlim(0, xmax)
ax.set_ylim(-750, -100)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

fig.tight_layout()
fname = f"../../results/paper-plots/{env_name}_attacked_rewards.pdf"
plt.savefig(dirpath / fname)
plt.close()


# ------------------------------------------------------------------------------
#
#                       Perturbed reward vs. Lipschitz bound
#
# ------------------------------------------------------------------------------

def plot_perturbed_reward_lip(perturbation, results, env_name):
    
    # Format the error bars
    ecolor = "grey"
    elinewidth = 0.8
    
    fig, ax = plt.subplots(figsize=(6.2,5.8))
    for network in results:
        
        index = np.isclose(results[network]["attack_sizes"], perturbation)
        x = results[network]["lip_mean"]
        y = results[network]["rewards"][index]
        x_err = results[network]["lip_stdev"]
        y_err = results[network]["stdev"][index]
        
        marker = "o" if network == "Unconstrained" else "s"
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=marker, 
                    label=network, ecolor=ecolor, 
                    elinewidth=elinewidth, markersize=8)

    ps = str(perturbation)
    if do_title: ax.set_title(f"Rewards at adversarial attack = {ps} ({env_name})")
    ax.set_ylim(-750, -100.0)
    ax.set_xlabel("Lipschitz lower bound $\\underline\\gamma$")
    ax.set_ylabel("Reward")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.tight_layout()
    fname = f"../../results/paper-plots/{env_name}_attacked_rewards_lip_a{ps}.pdf"
    plt.savefig(dirpath / fname)
    plt.close()
    
for p in perts_to_plot:
    if isinstance(p, np.float32):
        p = np.round(p,2)
    plot_perturbed_reward_lip(p, results, env_name)
