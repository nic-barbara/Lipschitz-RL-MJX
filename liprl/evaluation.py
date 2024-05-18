"""
This file contains an adaptation of some functions in
`brax.training.acting`.
"""

import jax
import jax.numpy as jnp
import optax

from brax import envs
from brax.training.types import Metrics, Policy, PRNGKey
from liprl.networks.utils import l2_norm

State = envs.State
Env = envs.Env


def track_results(state: State, ctrl: jnp.ndarray, eps: jnp.ndarray=0) -> dict:
    return {
        "ctrl": ctrl,
        "qpos": state.pipeline_state.q,
        "qvel": state.pipeline_state.qd,
        "obs": state.obs,
        "reward": state.reward,
        "done": state.done,
        "metrics": state.metrics,
        "attacks": eps,
    }


def swap_batches_time(results):
    """Swap an array with dims ordered as (time, batches, ...)
    to (batches, time, ...) for nice results processing."""
    for key in results:
        if isinstance(results[key], dict):
            swap_batches_time(results[key])
        else:
            
            # Ignore anything that's one-dimensional
            if results[key].ndim == 1:
                continue
            
            # Adversarial attack code returns (episodes/attack_episodes, 
            # attack_epochs, *obs.shape) and needs to be concatenated across
            # the attack runs
            if results[key].ndim > 3:
                results[key] = jnp.concatenate(results[key])
            results[key] = jnp.swapaxes(results[key], 0, 1)


class Evaluator:
    """Class to run evaluations over many environments, with option
    to introduce sample delays into the closed-loop rollouts."""

    def __init__(self, 
                 env: envs.Env,
                 policy,
                 num_eval_envs: int,
                 episode_length: int, 
                 seed: int = 0,
                 log_states: bool = False,
                 randomization_fn = None):
        """Init.

        Args:
        
        env: Environment to run evals on (we do the batching for you in here).
        policy: The final, trained policy function.
        num_eval_envs: Each env will run 1 episode in parallel for each eval.
        episode_length: Maximum length of an episode.
        seed: Seed for RNG key (default 0).
        log_states: Whether to keep track of all states/metrics through time
                    (default False).
        randomization_fn: See `ppo.train` for details (default None).
        """
        
        if randomization_fn:
            raise NotImplementedError("Dynamics randomisation not implemented here yet.")
        
        # Store information
        self.num_eval_envs = num_eval_envs
        self.episode_length = episode_length
        self.randomization_fn = randomization_fn
        
        # Set up batch environment
        self.key = jax.random.PRNGKey(seed)
        self.eval_env = self.batch_env(env)
        
        # Choose state-tracking function
        if log_states:
            self.track_results = track_results 
        else:
            self.track_results = lambda x,y: None
        
        # Function to do the unrolling
        def eval_unroll(key: PRNGKey, sample_delay) -> tuple[Metrics, dict]:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = self.eval_env.reset(reset_keys)
            final_state, rewards = self.generate_delayed_unroll(
                self.eval_env, eval_first_state, policy, episode_length, sample_delay
            )
            return final_state, rewards
            
        self.eval_unroll = jax.jit(eval_unroll, static_argnums=1)
        
    def run_evaluation(self, sample_delay:int=0):
        """Run one epoch of evaluation and process results. Option
        to include sample delays in the rollout.
        
        Returns aggregated results at the final state and also
        results through time if `log_states` was `True` on init.
        """
        return self._run_evaluation(sample_delay)
    
    def _run_evaluation(self, sample_delay=0) -> tuple[Metrics, dict]:
        
        # Closed-loop rollout
        self.key, unroll_key = jax.random.split(self.key)
        final_state, results = self.eval_unroll(unroll_key, sample_delay)
        
        # Aggregates metrics of the final state
        eval_metrics = final_state.info['eval_metrics']
        
        metrics = {}
        suffixes = ["", "_std", "_max", "_min"]
        eval_funcs = [jnp.mean, jnp.std, jnp.max, jnp.min]
        
        for i in range(len(eval_funcs)):
            fn = eval_funcs[i]
            suffix = suffixes[i]
            metrics.update({f'eval/episode_{name}{suffix}': (fn(value))
                            for name, value in eval_metrics.episode_metrics.items()})
            
        # Convert results through time to (batches, time, data...)
        if not results == None:
            swap_batches_time(results)
        return metrics, results
       
    def batch_env(self, env: envs.Env):
        """Set up a batch environment with BRAX.
        
        If you want domain randomization, check out the original
        code in ppo.train where they wrap the eval_env.
        """
        eval_env = envs.training.wrap(env, episode_length=self.episode_length)
        return envs.training.EvalWrapper(eval_env)
    
    def generate_delayed_unroll(self, 
                                env: Env, 
                                env_state: State, 
                                policy: Policy,
                                unroll_length: int,
                                sample_delay: int = 0) -> tuple[State, dict]:
        """Collect closed-loop trajectories of given `unroll_length` with
        sample delays if specified on initialisation."""
        
        @jax.jit
        def f(carry, unused_t):
            
            # Extract current state and RNG keys
            state, current_key, obs_buff = carry
            current_key, next_key = jax.random.split(current_key)
            
            # Track observations and introduce sample delay if required
            obs_buff = (state.obs, *obs_buff[0:-1])
            obs = obs_buff[-1]
            
            # Evaluate policy and environment
            ctrl, _ = policy(obs, current_key)
            nstate = env.step(state, ctrl)
        
            return (nstate, next_key, obs_buff), self.track_results(state, ctrl)

        # Initial state and observation buffer
        obs_buff = (env_state.obs,) * (sample_delay + 1)
        init_state = (env_state, self.key, obs_buff)
        
        # Roll out the policy over the full episode
        final_state, results = jax.lax.scan(f, init_state, (), length=unroll_length)
        return final_state[0], results


class AttackedEvaluator(Evaluator):
    """Class to run evaluations over many environments with
    adversarial attacks in the loop."""
    
    def __init__(self, 
                 env: envs.Env,
                 policy,
                 num_eval_envs: int,
                 episode_length: int,
                 attack_horizon: int = 100,
                 attack_epochs: int = 0,
                 attack_lr: jnp.float32 = 8e-3,
                 seed: int = 0,
                 log_states: bool = False,
                 randomization_fn = None):
        """Init.

        Args:
        
        env: Environment to run evals on (we do the batching for you in here).
        policy: The final, trained policy function.
        num_eval_envs: Each env will run 1 episode in parallel for each eval.
        episode_length: Maximum length of an episode.
        attack_horizon: Number of timesteps to evaluate the reward over for the attacker
                        (default 100). Episode length must be an integer multiple of
                        the attack horizon.
        attack_epochs: Number of gradient descent epochs for the attacker (default 1).
        attack_lr: Learning rate for the attacker (default 8e-3).
        seed: Seed for RNG key (default 0).
        log_states: Whether to keep track of all states/metrics through time
                    (default False).
        randomization_fn: See `ppo.train` for details (default None).
        """
        
        if randomization_fn:
            raise NotImplementedError("Dynamics randomisation not implemented here yet.")
        
        if not episode_length % attack_horizon == 0:
            raise ValueError("Episode length must be an integer multiple of the attack horizon.")
        
        # Store information
        self.num_eval_envs = num_eval_envs
        self.episode_length = episode_length
        self.randomization_fn = randomization_fn
        self.attack_epochs = attack_epochs
        self.attack_horizon = attack_horizon
        self.attack_lr = attack_lr
        self.sample_delay = 0
        
        # Set up batch environment
        self.key = jax.random.PRNGKey(seed)
        self.eval_env = self.batch_env(env)
        
        # Choose state-tracking function
        if log_states:
            self.track_results = track_results 
        else:
            self.track_results = lambda x,y,z=0: None
        
        # Define attacker loss function
        def perturbed_unroll(eps: jnp.ndarray, state: State, attack_size: jnp.float32):
            final_state, results = self.generate_perturbed_unroll(
                self.eval_env, state, policy, attack_horizon, eps, attack_size
            )
            return final_state, results
            
        def attacker_loss(eps: jnp.ndarray, state: State, attack_size: jnp.float32):
            final_state, _ = self.generate_perturbed_unroll(
                self.eval_env, state, policy, attack_horizon, eps, attack_size
            )
            return jnp.mean(final_state.reward)
        
        def eval_unroll(key: PRNGKey, attack_size: jnp.float32) -> State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = self.eval_env.reset(reset_keys)
            final_state, results = self.generate_attacked_unroll(
                eval_first_state, attack_size
            )
            return final_state, results
        
        # JIT everything
        self.attacker_loss = jax.jit(attacker_loss)
        self.perturbed_unroll = jax.jit(perturbed_unroll)
        self.eval_unroll = jax.jit(eval_unroll)
        
        if self.attack_epochs > 0:
            self.attacker_loss_grad = jax.jit(jax.grad(self.attacker_loss))
        else:
            self.attacker_loss_grad = lambda a,b,c: None

    def run_evaluation(self, attack_size:jnp.float32=0.0):
        """Run one epoch of evaluation and process results. Option
        to include adersarial attacks in the rollout.
        
        Returns aggregated results at the final state and also
        results through time if `log_states` was `True` on init.
        """
        return self._run_evaluation(attack_size)
    
    def generate_attacked_unroll(self, env_state: State, 
                                 attack_size: jnp.float32 = 0.0) -> State:
        """Collect closed-loop trajectories of a given `episode_length` subject
        to adversarial attacks with magnitude `attack_size`."""
        
        loss_grads = lambda eps, x: self.attacker_loss_grad(eps, x, attack_size)
        
        @jax.jit
        def f(carry, unused_t):
            
            state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            
            # Initialise perturbation
            obs0 = state.obs
            eps = jax.random.normal(current_key, (self.attack_horizon, *obs0.shape))
            eps = attack_size * self.normalise_attack(eps)
            
            # Train for a few epochs if required
            if self.attack_epochs > 0:
                opt = optax.adam(self.attack_lr)
                opt_state = opt.init(eps)
                
                def train_loop(carry, unused_t1):
                    eps, opt_state = carry
                    # See https://github.com/nic-barbara/Lipschitz-RL/issues/38
                    grad_value = jnp.nan_to_num(loss_grads(eps, state))
                    updates, new_opt_state = opt.update(grad_value, opt_state)
                    eps = optax.apply_updates(eps, updates)
                    return (eps, new_opt_state), None
                
                (eps, _), _ = jax.lax.scan(train_loop, (eps, opt_state), (), 
                                        length=self.attack_epochs)
                
            # Apply the trained perturbation, add to results if required
            nstate, results = self.perturbed_unroll(eps, state, attack_size)
            return (nstate, next_key), results
        
        n = self.episode_length // self.attack_horizon
        init_state = (env_state, self.key)
        final_state, results = jax.lax.scan(f, init_state, (), length=n)
        return final_state[0], results
    
    def generate_perturbed_unroll(self, 
                                  env: Env, 
                                  env_state: State, 
                                  policy: Policy,
                                  unroll_length: int,
                                  eps: jnp.ndarray,
                                  attack_size: jnp.float32) -> State:
        """Collect closed-loop trajectories of a given `unroll_length` subject
        to additive perturbations `eps` scaled by `attack_size` on the observations.
        
        Perturbations `eps` should have shape `(unroll_length, *env_state.obs.shape)`.
        """

        @jax.jit
        def f(carry, eps_t):
            
            # Extract current state and RNG keys
            state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            
            # Evaluate with perturbed observations
            obs = state.obs + eps_t
            ctrl, _ = policy(obs, current_key)
            nstate = env.step(state, ctrl)
        
            return (nstate, next_key), self.track_results(state, ctrl, eps_t)

        # Roll out the policy over the full episode
        init_state = (env_state, self.key)
        eps = attack_size * self.normalise_attack(eps)
        final_state, results = jax.lax.scan(f, init_state, eps, length=unroll_length)
        return final_state[0], results
    
    def normalise_attack(self, eps: jnp.ndarray) -> jnp.ndarray:
        return eps / l2_norm(eps, axis=2, keepdims=True)
