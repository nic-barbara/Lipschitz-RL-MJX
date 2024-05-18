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
perts_to_plot = [2] # Which delay values for scatter plot


def get_perturbed_reward_data(data):
    """Extract interesting data to plot for a given network type."""
    
    # Perturbed reward data
    r = np.array([d["delayed_rewards"]["rewards"] for d in data])
    
    # Perturbations and Lipschitz bounds
    perturbations = data[0]["sample_delays"]
    lips = np.array([d["estimated_lipschitz"] for d in data])
    
    out = {
        "sample_delays": np.array(perturbations),
        "rewards": r.mean(axis=0),
        "stdev": r.std(axis=0),
        "lip_mean": np.array([lips.mean()]),
        "lip_stdev": np.array([lips.std()]),
    }
    return out

def hex_to_rgb(hex_color):
    """Useful for plotting"""
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    return rgb_color


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
#                            Perturbed reward curves
#
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2,5.8))
xmax = 0
bar_width, i = 0.1, 0

# Format the colours nicely
ecolor = "grey"
elinewidth = 0.8
alpha = 0.6
buffer = 1.15
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [hex_to_rgb(c) for c in colors]

for network in results:
    x = results[network]["sample_delays"]
    y = results[network]["rewards"]
    y_err = results[network]["stdev"]
    
    bar = ax.bar(x + (i-2) * bar_width * buffer, y+750, bar_width, 
                 bottom=-750, 
                 label=network, 
                 linewidth=1.2,
                 edgecolor=colors[i], 
                 facecolor=(*colors[i], alpha))
    
    marker = "."
    ax.errorbar(x + (i-2) * bar_width * buffer, y, 
                yerr=y_err, 
                fmt=marker, 
                ecolor=ecolor, 
                elinewidth=elinewidth, 
                markersize=1,
                color=colors[i])
    
    i += 1
    xmax = np.max((max(x), xmax))

ax.set_xlabel("Delay (samples)")
ax.set_ylabel("Reward")
ax.set_ylim(-750, -100.0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

fig.tight_layout()
fname = f"../../results/paper-plots/{env_name}_delayed_rewards.pdf"
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
        
        index = np.isclose(results[network]["sample_delays"], perturbation)
        x = results[network]["lip_mean"]
        y = results[network]["rewards"][index]
        x_err = results[network]["lip_stdev"]
        y_err = results[network]["stdev"][index]
        
        marker = "o" if network == "Unconstrained" else "s"
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=marker, 
                    label=network, ecolor=ecolor, 
                    elinewidth=elinewidth, markersize=8)

    ps = str(perturbation)
    if do_title: ax.set_title(f"Rewards at sample delay = {ps} ({env_name})")
    ax.set_ylim(-750, -100.0)
    ax.set_xlabel("Lipschitz lower bound $\\underline\\gamma$")
    ax.set_ylabel("Reward")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.tight_layout()
    fname = f"../../results/paper-plots/{env_name}_delayed_rewards_lip_d{ps}.pdf"
    plt.savefig(dirpath / fname)
    plt.close()
    
for p in perts_to_plot:
    if isinstance(p, np.float32):
        p = np.round(p,2)
    plot_perturbed_reward_lip(p, results, env_name)
