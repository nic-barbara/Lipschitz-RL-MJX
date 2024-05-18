import liprl.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from liprl.experiment import Experiment
from pathlib import Path


def train_hyperparam_tuning(env_name, 
                            default_config, 
                            tuning_config, 
                            verbose=True, 
                            keep_best_params=False,
                            do_plots=False, 
                            env_subdirpath=""):
    
    if env_subdirpath == "":
        env_subdirpath = "tuning/"
        
    # Set the file path
    dirpath = Path(__file__).resolve().parent
    default_config["fpath"] = str(
        dirpath / "../results/" / env_name / env_subdirpath)
    
    # Run the default configuration
    default_config["version"] = _get_version(default_config)
    if verbose:
        print("\n############# Tuning parameters for {} network on {}: #############"
              .format(default_config["network"], env_name))
        print("\nTraining " + default_config["version"] + " (default):")
    
    e = Experiment(env_name)
    e.init_config(default_config)
    e.train(verbose=verbose, keep_best_params=keep_best_params)
    if do_plots: 
        utils.startup_plotting()
        e.plot_rewards()
    
    # Loop through other variants one at a time (not full grid search)
    for key in tuning_config:
        config = deepcopy(default_config)
        
        # Only loop through Lipschitz bounds for LBDN
        if (key == "gamma") and (config["network"] == "mlp"):
            continue
        
        for value in tuning_config[key]:
            config[key] = value
            
            # Don't double-up on the default params
            config["version"] = _get_version(config)
            if not config["version"] == default_config["version"]:

                config["version"] = _get_version(config)
                if verbose:
                    print("\nTraining " + config["version"] + ":")
                    
                # Train with this config and plot rewards
                e = Experiment(env_name)
                e.init_config(config)
                e.train(verbose=verbose, keep_best_params=keep_best_params)
                if do_plots: 
                    e.plot_rewards()


def _get_version(config: dict) -> str:
    v = "{}_n{}_r{}_u{}_d{}_lr{}".format(
        config["activation"],
        config["num_updates_per_batch"],
        config["reward_scaling"],
        config["unroll_length"],
        config["discounting"],
        config["learning_rate"],
    )
    if not (config["network"] == "mlp"):
        v += "_g{}".format(config["gamma"])
    v += "_s{}".format(config["seed"])
    return v


class HyperparamTuningResults:
    """
    What this class does/should be used for:
        1. New figure for each type of network
        2. New subplot for each tuning parameter
        3. Plot a reward curve for each unique value of that hyperparameter
        4. Observe results
        
    Create an instance and use the `plot_rewards()` method.
    """
    
    def __init__(self, env_name, tuning_params, dirname="tuning"):
        
        # Get all file names in the tuning directory
        self.env_name = env_name
        dirpath = Path(__file__).resolve().parent
        self.fpath = dirpath / str("../results/" + env_name + "/" + dirname)
        files = [f for f in self.fpath.iterdir() if (
            f.is_file() and not (f.suffix == ".pdf"))]
        
        # Read files and store configs and metrics
        self.configs_metrics = []
        for f in files:
            data = utils.load_params(f)
            self.configs_metrics.append((data[2], data[3]))
            
        # Get (unique) network types and tuning params
        self.network_types = utils.lunique([c[0]["network"] for c in self.configs_metrics])
        self.tuning_params = tuning_params
        
        # Nice plot formatting
        utils.startup_plotting()
    
    def plot_rewards(self):
        """
        Create a figure for each network type with subplots
        showing reward curves for each hyperparameter combination.
        """
        
        for network in self.network_types:
            self._plot_network_rewards(network)
    
    def _plot_network_rewards(self, network):
        """
        Single figure for a given network type with subplots
        showing reward curves for each hyperparameter combination.
        """
        
        # Ignore Lipschitz bounds for the MLP (not tunable)
        tuning_params = self.tuning_params
        if network == "mlp":
            tuning_params = [t for t in tuning_params if not t == "gamma"]
            
        # Only consider configs for this network
        network_data = [c for c in self.configs_metrics 
                        if c[0]["network"] == network]
            
        # Set up figure and title
        subplot_dims = self._get_subplot_shape(tuning_params)
        fsize = self._get_figsize(tuning_params)
        fig, axs = plt.subplots(*subplot_dims, sharey=True, figsize=fsize)
        fig.suptitle("Hyperparameter tuning for {} ({})".format(
            network.upper(), self.env_name))
        
        # Construct each subplot
        for i in range(len(tuning_params)):
            if len(tuning_params) == 1:
                ax = axs
            elif len(tuning_params) == 2:
                ax = axs[i]
            else:
                ax = axs[i % 2, i // 2]
            self._plot_hyperparam_rewards(ax, tuning_params[i], network_data)
        
        # Remove any blank tiles
        if len(tuning_params) % 2 == 1 and len(tuning_params) > 1:
            axs[-1, -1].axis("off")

        # Add a big subplot with a common axes
        fig.add_subplot(frameon=False)
        ax = plt.gca()
        ax.tick_params(labelcolor='none', which='both', top=False, 
                        bottom=False, left=False, right=False)
        ax.grid(visible=False)
        ax.set_xlabel("Environment steps", labelpad=20.0)
        ax.set_ylabel("Reward", labelpad=20.0)
        
        # Format, save, close
        plt.tight_layout()
        plt.savefig(str(self.fpath / (self.env_name + 
                                      "_tuning_summary_" + 
                                      network + ".pdf")))
        plt.close()
    
    def _plot_hyperparam_rewards(self, ax, hyperparam, configs_metrics):
        """
        Single-axis plot of reward curves for a given hyperparameter.
        """
        
        # Find unique values of this hyperparam and all config dicts with
        # those values
        unique_hps = utils.lunique([c[0][hyperparam] for c in configs_metrics])
        data_with_unique_hps = [c for c in configs_metrics 
                                if (c[0][hyperparam] in unique_hps)]
        data_with_unique_hps.reverse()
        
        # Narrow it down to just the config dicts varying this hyperparam
        # (exclude any others that use a value of this hyperparam as default)
        unique_data = []
        if len(data_with_unique_hps) == 1:
            unique_data = data_with_unique_hps
        else:
            ignore_keys = ["version", "seed", hyperparam]
            match = False
            for i in range(len(data_with_unique_hps)):
                for j in range(i+1, len(data_with_unique_hps)):
                    if self._match_dicts(data_with_unique_hps[i][0],
                                        data_with_unique_hps[j][0],
                                        ignore_keys):
                        
                        if unique_data == []:
                            unique_data.append(data_with_unique_hps[i])
                        unique_data.append(data_with_unique_hps[j])
                        match = True
                if match: 
                    break
            
        # Do the plotting
        ax.set_title(hyperparam)
        for i in range(len(unique_data)):
            x = np.array(unique_data[i][1]["steps"])
            y = np.array(unique_data[i][1]["rewards"])
            y_err = np.array(unique_data[i][1]["stdev"])
            
            ax.plot(x, y, label="{}".format(unique_data[i][0][hyperparam]))
            ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
        ax.legend()
    
    def _get_subplot_shape(self, tuning_params):
        n = len(tuning_params)
        if n <= 2:
            return (1, n)
        return (2, (n + 1) // 2)
    
    def _get_figsize(self, tuning_params):
        n = len(tuning_params)
        if n == 2:
            return (8, 5)
        if n == 1:
            return (5, 5)
        return (1 + 3*((n + 1) // 2), 7)
    
    def _match_dicts(self, x: dict, y: dict, ignore_keys: list):
        """
        Returns true if two dictionaries are the same 
        aside from the keys listed in `ignore_keys`.
        """
        for key in x:
            if key in ignore_keys:
                continue
            if key not in y:
                return False
            if not x[key] == y[key]:
                return False
        return True
