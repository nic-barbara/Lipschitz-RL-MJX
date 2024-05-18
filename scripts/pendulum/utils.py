import numpy as np
import jax
import matplotlib.pyplot as plt

from liprl.analysis_tools import ExperimentAnalyser
from matplotlib import ticker


def order_networks(networks: list, lips: list):
    """Sort list of networks with MLP at start for plotting convenience."""
    index = ["mlp" in n for n in networks]
    mlp_index = np.arange(len(index))[index][0]
    networks.insert(0, networks.pop(mlp_index))
    lips.sort()
    return networks, lips


def _get_pendulum_observation_grid(theta, theta_d):
    xx, yy = np.meshgrid(theta, theta_d)
    states = np.vstack([xx.ravel(), yy.ravel()])
    obs = np.vstack((
        np.cos(states[0,:]),
        np.sin(states[0,:]),
        states[1,:])).T
    return obs, xx, yy


def _diff_same_dim(x):
    dx = np.diff(x)
    return np.hstack((x[:1], x[1:] - dx))


def _safe_norm_ax1(x, eps=1e-12):
    return np.max((np.linalg.norm(x, axis=1), 
                   eps*np.ones(x.shape[0])), axis=0)
    

def plot_pendulum_contours(policy, 
                           fname, 
                           npoints=400,
                           theta_max=3*np.pi/2, 
                           vel_max=8):
    
    # Create series of points to run the network over
    theta = np.linspace(-theta_max, theta_max, npoints)
    theta_d = np.linspace(-vel_max, vel_max, npoints)
    obs, xx, yy = _get_pendulum_observation_grid(theta, theta_d)

    # Evaluate the network
    rng = jax.random.PRNGKey(0)
    ctrl, _ = policy(obs, rng)
    ctrl = np.reshape(ctrl, (npoints, npoints))
    
    # Plot the response as a heat map
    colourset = plt.contourf(xx, yy, ctrl, cmap="hot", levels=20)
    cbar = plt.colorbar(colourset)
    plt.clim(-1, 1)
    cbar.ax.set_ylabel("Control torque (N.m)")
    
    # Format the figure
    plt.xlabel("Angular position (rad)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    

def plot_pendulum_lipschitz_contours(policy, 
                                     fname, 
                                     npoints=400,
                                     theta_max=3*np.pi/2, 
                                     vel_max=8,
                                     max_clim=None):
    
    # Random seed setup
    rng = jax.random.PRNGKey(0)
    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
    
    # Do finite difference over four directions
    theta1 = np.linspace(-theta_max, theta_max, npoints)
    theta_d1 = np.linspace(-vel_max, vel_max, npoints)
    theta2 = _diff_same_dim(theta1)
    theta_d2 = _diff_same_dim(theta_d1)
    
    # Create a bunch of observation grids (first one is for plotting)
    obs11, xx, yy = _get_pendulum_observation_grid(theta1, theta_d1)
    obs12, _, _ = _get_pendulum_observation_grid(theta1, theta_d2)
    obs21, _, _ = _get_pendulum_observation_grid(theta2, theta_d1)
    obs22, _, _ = _get_pendulum_observation_grid(theta2, theta_d2)
    
    # Compute controls on each grid
    def _get_ctrl(obs, rng):
        ctrl, _ = policy(obs, rng)
        return ctrl
    
    u11 = _get_ctrl(obs11, rng1)
    u12 = _get_ctrl(obs12, rng2)
    u21 = _get_ctrl(obs21, rng3)
    u22 = _get_ctrl(obs22, rng4)
    
    # Estimate local Lipschitz bound
    g1 = _safe_norm_ax1(u11 - u12) / _safe_norm_ax1(obs11 - obs12)
    g2 = _safe_norm_ax1(u11 - u21) / _safe_norm_ax1(obs11 - obs21)
    g3 = _safe_norm_ax1(u11 - u22) / _safe_norm_ax1(obs11 - obs22)
    gamma = np.max((g1,g2,g3), axis=0)
    gamma = np.reshape(gamma, (npoints, npoints))
    
    # Plot the result as a heat map
    colourset = plt.contourf(xx, yy, gamma, cmap="hot", levels=20)
    cbar = plt.colorbar(colourset)
    if max_clim: 
        plt.clim(0, max_clim) # Allows for common colour scale across figs
    cbar.ax.set_ylabel("Local Lipschitz bound")
    
    # Format the figure
    plt.xlabel("Angular position (rad)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_pendulum_timedomain_sims(e: ExperimentAnalyser, 
                                  fname,
                                  theta_max=3*np.pi/2, 
                                  vel_max=6,
                                  num_envs=128,
                                  n_steps=150,
                                  perturbation=0,
                                  attack_horizon: int = 0,
                                  attack_epochs: int = 0):
    
    # Set limits on initial states
    e.env._reset_pos_range=theta_max
    e.env._reset_vel_range=vel_max
    
    # Compute trajectories
    e.setup_batch_eval(n_steps=n_steps, num_envs=num_envs, log_states=True,
                       attack_horizon=attack_horizon, attack_epochs=attack_epochs)
    _, results = e.batch_eval_mjx(perturbation)
    theta = np.squeeze(results["qpos"]).T
    t = np.arange(0, n_steps) * e.env.dt
    
    # Add some red lines for the target states
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(t, np.zeros(t.shape), 'r--', linewidth=1.0)
    ax.plot(t, 2*np.ones(t.shape), 'r--', linewidth=1.0)
    ax.plot(t, -2*np.ones(t.shape), 'r--', linewidth=1.0)
    
    # Plot the norm of the states in the time domain
    ax.plot(t, theta / np.pi, color="grey", linewidth=0.25, alpha=0.25)
    
    # Format the figure
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pendulum angle (rad)")
    ax.set_xlim(t.min(), t.max())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_set_pi_axes))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()


def _set_pi_axes(val, pos):
    if val == 0:
        return '0'
    if val == 1:
        return '$\pi$'
    if val == -1:
        return '-$\pi$'
    return '{:.0g}$\pi$'.format(val)
