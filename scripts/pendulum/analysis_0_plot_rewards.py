import liprl.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from liprl.utils import startup_plotting
from pathlib import Path
from utils import order_networks

# Plotting choices
do_title = False
fsize = 20
startup_plotting(font_size=fsize)


def get_reward_data(data):
    r = np.array([d["rewards"] for d in data])
    out = {
        "steps": data[0]["steps"],
        "rewards": r.mean(axis=0),
        "stdev": r.std(axis=0),
    }
    return out

# Get all file names
env_name = "pendulum"
dirpath = Path(__file__).resolve().parent
fpath = dirpath / f"../../results/{env_name}/trained-models"
files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]

# Read files and store configs and metrics
configs_metrics = []
for f in files:
    data = utils.load_params(f)
    configs_metrics.append((data[2], data[3]))
    
# Get unique network types
network_types = utils.lunique([c[0]["network"] for c in configs_metrics])
lipschitz_bounds = utils.lunique([c[0]["gamma"] for c in configs_metrics 
                                  if not c[0]["network"] == "mlp"])
network_types, lipschitz_bounds = order_networks(network_types, lipschitz_bounds)

# Loop through networks and store data
results = {}
for network in network_types:
    if network == "mlp":
        name_ = "Unconstrained"
        data = [c[1] for c in configs_metrics if c[0]["network"] == network]
        results[name_] = get_reward_data(data)
        continue
    for g in lipschitz_bounds:
        name_ = "Lipschitz" + " ($\gamma = {:.0f}$)".format(g)
        data = [c[1] for c in configs_metrics 
                if (c[0]["network"] == network) and (c[0]["gamma"] == g)]
        results[name_] = get_reward_data(data)

# Create the plot
fig, ax = plt.subplots()
xmax = 0
for network in results:
    x = np.array(results[network]["steps"]) / 1e6
    y = -np.array(results[network]["rewards"])
    y_err = -np.array(results[network]["stdev"])
    
    lstyle = "dotted" if network == "Unconstrained" else "solid"
    
    ax.plot(x, y, label=network, linestyle=lstyle)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)

if do_title: ax.set_title("Tuned models on {} environment".format(env_name))
ax.set_xlabel("Environment steps ($\\times 10^6$)")
ax.set_ylabel("Reward")
ax.set_xlim(0, 20)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_yticks([1e2,1e3],["$-10^{2}$","$-10^{3}$"])
ax.legend(loc="lower right")

fig.tight_layout()
plt.savefig(dirpath / f"../../results/paper-plots/{env_name}_tuned_rewards.pdf")
plt.close()
