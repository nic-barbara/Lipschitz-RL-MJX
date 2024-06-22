import jax

from liprl.evaluation import Evaluator, AttackedEvaluator
from liprl.experiment import Experiment


class ExperimentAnalyser(Experiment):
    
    def __init__(self, 
                 env_name: str, 
                 full_fpath: str = "", 
                 deterministic: bool = True,
                 **kwargs):
        
        # Load the experiment
        super().__init__(env_name)
        self.load_config(full_fpath=full_fpath, deterministic=deterministic, **kwargs)
        
        # Check if correctly loaded
        self._check_config_loaded()
        self._check_trained()

    def setup_batch_eval(self,
                         seed: int = 0,
                         n_steps: int = 100,
                         num_envs: int = 128,
                         log_states: bool = False,
                         attack_horizon: int = 0,
                         attack_epochs: int = 0):
        """Set up for batch evaluations. 
        
        Set `attack_horizon` and `attack_epochs` to set up a batch evaluation
        with adversarial attacks.
        
        Args:
        
        seed: the random seed (default 0).
        n_steps: number of steps to simulate over (default 100).
        num_envs: number of environments to run in parallel (default 128).
        log_states: whether to log states/metrics at each timestep (default False).
        attack_horizon: Number of timesteps to evaluate the reward over for the attacker
                        (default 5).
        attack_epochs: Number of gradient descent epochs for the attacker (default 1).
        """
        if attack_horizon == 0:
            self.attacked = False
            self.evaluator = Evaluator(self.env,
                                       self.policy,
                                       num_eval_envs=num_envs,
                                       episode_length=n_steps,
                                       seed=seed,
                                       log_states=log_states)
        else:
            self.attacked = True
            self.evaluator = AttackedEvaluator(self.env,
                                               self.policy,
                                               num_eval_envs=num_envs,
                                               episode_length=n_steps,
                                               seed=seed,
                                               log_states=log_states,
                                               attack_epochs=attack_epochs,
                                               attack_horizon=attack_horizon)


    def batch_eval_mjx(self, perturbation=0):
        """Evaluate a policy in MJX over a batch of environments.
        
        Args:
        
        perturbation: The perturbation to apply to the closed-loop system.
                      If `setup_batch_eval` was set up for adversarial attacks,
                      this is the attack size. Otherwise, it's the sample delay.
        """
        final_state, results = self.evaluator.run_evaluation(perturbation)
        return final_state, results
    
    def eval_perturbed_rewards_mjx(self, perturbations=[0]):
        """
        Evaluate a trained policy on an MJX environment over a range
        of perturbations and return aggregated metrics for the final state.
        
        Args:
        
        perturbation: The perturbation to apply to the closed-loop system.
                      If `setup_batch_eval` was set up for adversarial attacks,
                      this is the attack size. Otherwise, it's the sample delay.
        """
        
        # Choose function for evaluation
        def vec_eval(x):
            final_state, _ = self.batch_eval_mjx(x)
            return final_state
        
        # TODO: Cannot currently vmap the delays. This is because we use
        # the delay for indexing, and it's an argument. It needs to be an
        # int, but calling int() on a jax array implementation in vmap doesn't
        # seem to work...
        if self.attacked:
            reward_data = jax.vmap(vec_eval)(perturbations)
        else:
            metrics = [vec_eval(d) for d in perturbations]
            
            # Store in a nicer format for analysis
            reward_data = {}
            for key in metrics[0]:
                reward_data[key] = []
                for m in metrics:
                    reward_data[key].append(m[key])
            
        # Store in a nicer format for analysis
        reward_data["rewards"] = reward_data.pop("eval/episode_reward")
        reward_data["stdev"] = reward_data.pop("eval/episode_reward_std")
        reward_data["max"] = reward_data.pop("eval/episode_reward_max")
        reward_data["min"] = reward_data.pop("eval/episode_reward_min")
            
        return reward_data
