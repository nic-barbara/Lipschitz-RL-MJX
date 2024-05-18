import brax.envs
import functools
import jax
import liprl
import liprl.utils as utils
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np

from brax.training.acme import running_statistics
from datetime import datetime
from flax import linen
from jax import numpy as jnp
from liprl.ppo import networks as ppo_networks
from liprl.ppo import train as ppo
from mujoco import mjx
from pathlib import Path


class Experiment():
    """Class to train and test a policy with MJX/MuJoCo.
    The intended workflow is as follows:
    
        # Train a model
        e = Experiment("<env_name>")
        e.init_config(config_dict)
        e.train()
        e.save()
    
        # Load and evaluate a model
        e = Experiment("<env_name>")
        e.load_config(fpath)
        e.plot_rewards()
        e.eval_mjx()
        e.eval_mujoco()
        
    See also `liprl.analysis_tools` for `ExperimentAnalyser` - a subclass
    of `Experiment` for analysing trained policies on an environment.
    """
    
    def __init__(self, env_name):

        self.config = {}
        self.params = ()
        self.metrics = {
            "rewards": [], 
            "stdev": [],
            "steps": [],
            "times": [],
        }
        self.best_reward = -jnp.inf
        
        self.env_name = env_name
        self.fpath = Path("")
        self.fname_prefix = ""
        self.env = None
        self.policy = None
    
    def init_config(self, config: dict, **kwargs):
        """Config dictionary should contain the following:
        
            fpath: (optional) path to directory in which to save/load results.
                   Must be a string object.
            version: Experiment version number.
            gamma: Upper bound on the Lipschitz constant.
            train_lipschitz: Make the Lipschitz constant trainable.
            policy_sizes: tuple of hidden layer sizes for policy network.
            value_sizes: tuple of hidden layer sizes for value network.
            activation: string indicating the activation function.
        
        ...and all keyword arguments to `liprl.ppo.train` should also be included.
        See `_set_default_config` for default values.
        """
        
        # Set default config, replace with user inputs
        self._set_default_config()
        for key in config:
            self.config[key] = config[key]
        
        # Choose the environment
        self.env = brax.envs.get_environment(self.env_name, backend="mjx", **kwargs)
        self.config["n_obs"] = self.env.observation_size
        self.config["n_act"] = self.env.action_size
        self.config["env_name"] = self.env_name
        
        # Change the solver to speed up & support backpropagation (iters = 1)
        # See: https://github.com/google-deepmind/mujoco/issues/1182#issuecomment-1823411911
        solver_options = {
            'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
            'opt.iterations': 1,
            'opt.ls_iterations': 4,
        }
        if not(self.env_name in ["pusher", "halfcheetah", "humanoidstandup"]):
            self.env.sys = self.env.sys.tree_replace(solver_options)
            self.config["solver_options"] = solver_options
            
        # Directory to save results
        if "fpath" not in self.config.keys():
            config["fpath"] = self._get_default_fpath()
        self.fpath = Path(config["fpath"])
        self.fname_prefix = str(self.fpath) + (
            "/" + self.env_name + "_" + self.config["network"])
            
        if not self.fpath.exists():
            self.fpath.mkdir(parents=True)
        
    def train(self, verbose=True, do_save=True, deterministic=True, keep_best_params=False):
        """
        Train a network with PPO on the environment.
        
        Setting `deterministic=True` does **not** change the training. It
        just ensures the final policy is only the trained network, and
        not the network + Gaussian white noise on the outputs used for PPO.
        
        Setting `keep_best_params=True` stores the model params with the best test
        reward during training. Useful if an environment is plagued by catastrophic
        unlearning, which is painfully common in PPO.
        """
                
        # Create training function with defaults
        train_fn = functools.partial(
            ppo.train,
            num_evals=self.config["num_evals"],
            num_timesteps=self.config["num_timesteps"],
            episode_length=self.config["episode_length"],
            num_envs=self.config["num_envs"],
            learning_rate=self.config["learning_rate"],
            entropy_cost=self.config["entropy_cost"],
            discounting=self.config["discounting"],
            unroll_length=self.config["unroll_length"],
            batch_size=self.config["batch_size"],
            num_minibatches=self.config["num_minibatches"],
            num_updates_per_batch=self.config["num_updates_per_batch"],
            normalize_observations=self.config["normalize_observations"],
            seed=self.config["seed"],
            reward_scaling=self.config["reward_scaling"],
            normalize_advantage=self.config["normalize_advantage"],
            network_factory=self._network_factory(),
            num_eval_envs=128,
            deterministic_eval=True,
        )
        
        # Log rewards and print if required. Also track best params during training
        self.metrics["times"].append(datetime.now())
        self.config["keep_best_params"] = keep_best_params
        def progress(num_steps, metrics, params):
            self.metrics["times"].append(datetime.now())
            self.metrics["steps"].append(num_steps)
            self.metrics["rewards"].append(metrics["eval/episode_reward"])
            self.metrics["stdev"].append(metrics["eval/episode_reward_std"])
            
            if keep_best_params:
                self._update_best_params(params, metrics)
        
            if verbose:
                print("step: {} \t reward: {:.2f} \t stdev: {:.2f} \t time: {}".format(
                    num_steps, 
                    metrics["eval/episode_reward"], 
                    metrics["eval/episode_reward_std"],
                    self.metrics["times"][-1],))
        
        # Train model and save
        _, params, metrics = train_fn(environment=self.env, progress_fn=progress)
        if keep_best_params:
            self._update_best_params(params, metrics)
        else:
            self.params = params
        self._load_policy(deterministic=deterministic)
        
        if verbose:
            times = self.metrics["times"]
            print(f'time to jit: {times[1] - times[0]}')
            print(f'time to train: {times[-1] - times[1]}')
            print(f'best reward: {self.best_reward}')
            
        if do_save:
            self.save()

    def save(self):
        """Save a trained policy."""
        fpath = self.fname_prefix + ("_model_" + self.config["version"])
        data = (
            *self.params, 
            self.config,
            self.metrics,
        )
        utils.save_params(fpath, data)

    def load_config(self, network="mlp", version="v0", fpath="",
                    full_fpath="", deterministic=True):
        """Load a config dict, trained policy, and results."""
        
        if not fpath:
            fpath = self._get_default_fpath()
            
        if not full_fpath:
            fname = fpath + (
                "/" + self.env_name + "_" + network + "_model_" + version)
        else:
            fname = full_fpath
            
        data = utils.load_params(fname)
        data[2]["fpath"] = fpath
        
        self.params = (data[0], data[1])
        self.metrics = data[3]
        self.init_config(data[2])
        self._load_policy(deterministic=deterministic)
        
    def plot_rewards(self, fname_prefix="", fname_suffix=""):
        """Plot rewards during training from a given policy."""
        
        if not fname_prefix:
            fname_prefix = self.fname_prefix
            
        if not fname_suffix:
            fname_suffix = "_rewards_" + self.config["version"] + ".pdf"
        
        x = np.array(self.metrics["steps"])
        y = np.array(self.metrics["rewards"])
        y_err = np.array(self.metrics["stdev"])
        
        _, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Reward")
        plt.tight_layout()
        plt.savefig(fname_prefix + fname_suffix)
        plt.close()
        
    def eval_mjx(self, n_steps=100, sample_delay=0, 
                 render_every=1, camera=-1, fname_prefix="", 
                 fname_suffix="", do_save=True):
        """Evaluate a trained policy on its MJX environment.
        
        Option to include a sample delay to observe its effect.
        """
        
        self._check_config_loaded()
        self._check_trained()
        
        # Jit the policy and environment
        jit_policy = jax.jit(self.policy)
        jit_reset = jax.jit(self.env.reset)
        jit_step = jax.jit(self.env.step)
        
        # Initialize the state and sample delay buffer
        rng = jax.random.PRNGKey(self.config["seed"])
        state = jit_reset(rng)
        rollout = [state.pipeline_state]
        obs_buff = (state.obs,) * (sample_delay + 1)
        
        # Record a trajectory
        for i in range(n_steps):
            
            obs_buff = (state.obs, *obs_buff[0:-1])
            obs = obs_buff[-1]
            
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_policy(obs, act_rng)
            
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)

            if state.done:
                break
            
        if not do_save: 
            return None
        
        if not fname_prefix:
            fname_prefix = self.fname_prefix
            
        if not fname_suffix:
            fname_suffix = "_video_mjx_" + self.config["version"]
            if sample_delay > 0:
                fname_suffix += f"_delay{sample_delay}"
            fname_suffix += ".mp4"
            
        media.write_video(
            fname_prefix + fname_suffix,
            self.env.render(rollout[::render_every], 
                            camera=camera, width=600, height=480),
            fps=1.0 / self.env.dt / render_every
        )

    def eval_mujoco(self, n_steps=100, render_every=1, camera=-1, 
                    fname_prefix="", fname_suffix=""):
        """Evaluate a trained policy on its MuJoCo environment."""
        
        self._check_config_loaded()
        self._check_trained()
        
        # Load a MuJoCo model and JIT the policy
        mj_model = self.env.sys.mj_model
        mj_data = mujoco.MjData(mj_model)
        jit_policy = jax.jit(self.policy)
        rng = jax.random.PRNGKey(self.config["seed"])
        
        # Handle MJX vs. MuJoCo syntax and setup sim
        mujoco_obs = utils.ObservationWrapper(self.env)
        renderer = mujoco.Renderer(mj_model, width=600, height=480)
        ctrl = jnp.zeros(mj_model.nu)
        
        # Record a trajectory
        images = []
        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)

            # TODO: could probably find a more elegant solution here...
            d = utils.DataSpoofer(mjx.put_data(mj_model, mj_data))
            obs = mujoco_obs._get_obs(d, ctrl)
            ctrl, _ = jit_policy(obs, act_rng)

            mj_data.ctrl = ctrl
            for _ in range(self.env._n_frames):
                mujoco.mj_step(mj_model, mj_data)

            if i % render_every == 0:
                renderer.update_scene(mj_data, camera=camera)
                images.append(renderer.render())
        
        # Write video to file
        if not fname_prefix:
            fname_prefix = self.fname_prefix
            
        if fname_suffix == "":
            fname_suffix = "_video_mujoco_" + self.config["version"] + ".mp4"
            
        media.write_video(
            fname_prefix + fname_suffix,
            images,
            fps=1.0 / self.env.dt / render_every
        )
        renderer.close()
    
    def _network_factory(self):
        """Generate function to create networks"""
        
        return functools.partial(
            ppo_networks.make_ppo_networks,
            network=self.config["network"],
            policy_hidden_layer_sizes=self.config["policy_sizes"],
            value_hidden_layer_sizes=self.config["value_sizes"],
            activation=self._get_activation(self.config["activation"]),
            gamma=self.config["gamma"],
            trainable_lipschitz=self.config["train_lipschitz"],
        )
        
    def _load_policy(self, deterministic=True, return_policy=False):
        
        self._check_config_loaded()
        self._check_trained()
        
        # Set up input normalisation
        normalize_fn = lambda x, y: x
        if self.config["normalize_observations"]:
            normalize_fn = running_statistics.normalize
            
        # Re-construct the policy
        network_factory = self._network_factory()
        network_template = network_factory(
            self.config["n_obs"],
            self.config["n_act"],
            preprocess_observations_fn=normalize_fn,
        )
        network_constructor = ppo_networks.make_inference_fn(network_template)
        policy = network_constructor(self.params, deterministic=deterministic)
        if return_policy: 
            return policy
        self.policy = policy
    
    def _check_config_loaded(self):
        if not self.config:
            raise ImportError(
                "No config dict loaded. Run \
                `.load_config` first.")
    
    def _check_trained(self):
        if not self.params:
            raise SyntaxError(
                "No training params available. Run the \
                `.train()` method first.")
            
    def _get_activation(self, s: str):
        """Get activation function from flax.linen via string."""
        if s == "identity":
            return (lambda x: x)
        return eval("linen." + s)
    
    def _get_default_fpath(self):
        dirpath = Path(__file__).resolve().parent
        return str(dirpath / "../results/" / self.env_name)
    
    def _update_best_params(self, params, metrics):
        if metrics["eval/episode_reward"] >= self.best_reward:
            self.params = params
            self.best_reward = metrics["eval/episode_reward"]
        
    def _set_default_config(self):
        """Some reasonable hyperparameters for training.
        
        Network sizes here are the defaults used in Brax. Other hyperparameters
        are commonly-chosen parameters used to train Brax models, but will
        likely require tuning depending on the environment/network.
        """
        config = {
            "network": "mlp",
            "version": "v0",
            
            "policy_sizes": (32,) * 4,
            "value_sizes": (256,) * 5,
            "activation": "tanh",
            
            "num_timesteps": 50_000_000,
            "num_evals": 20,
            "reward_scaling": 1.0,
            "episode_length": 1000,         
            "normalize_observations": True,
            "normalize_advantage": False,
            "action_repeat": 1,
            "unroll_length": 10,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-3,
            "num_envs": 2048,
            "batch_size": 1024,
            "seed": 0,
            
            "gamma": 1.0,
            "train_lipschitz": False,
        }
        self.config = config
    