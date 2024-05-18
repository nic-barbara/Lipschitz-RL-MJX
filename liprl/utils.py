import numpy as np
import pickle

from cycler import cycler
from etils import epath
from inspect import signature
from matplotlib import pyplot as plt
from typing import Any


def load_params(path: str) -> Any:
    """Copied over from brax.io.model"""
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """
    Saves parameters in flax format.
    Copied over from brax.io.model
    """
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))
        
        
def lunique(x:list) -> list:
    """Return list of unique elements in a list."""
    y = np.unique(np.array(x))
    return list(y)


class DataSpoofer:
    """Wrapper for MjData or mjx.Data that allows accessing 
    d.qpos, d.qvel as d.q, d.qd."""
    
    def __init__(self, data):
        for k in data.__dict__:
            exec("self." + k + " = getattr(data, '" + k + "')")
        self.q = self.qpos
        self.qd = self.qvel
        

class ObservationWrapper:
    """Wrapper around MuJoCo environment to handle inputs for env._get_obs()"""
    
    def __init__(self, env):
        sig = signature(env._get_obs)
        self.observe_ctrl = len(sig.parameters) > 1
        self.env = env
        
    def _get_obs(self, data, ctrl):
        if self.observe_ctrl:
            obs = self.env._get_obs(data, ctrl)
        else:
            obs = self.env._get_obs(data)
        return obs


def startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True):
    """Edited from https://github.com/nackjaylor/formatting_tips-tricks/"""

    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
    
    # Default settings
    plt.rcParams.update({
        "lines.linewidth": line_width,
        
        "axes.grid" : True, 
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7", 
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]),

        "errorbar.capsize": 2.5,
        
        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,
        
        "legend.framealpha": 0.7,
        "legend.edgecolor": [1,1,1],
        
        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=0.8*font_size)
    plt.rc('ytick', labelsize=0.8*font_size)
    plt.rc('legend', fontsize=0.8*font_size)
