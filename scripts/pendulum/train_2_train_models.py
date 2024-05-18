from liprl.tuning import train_hyperparam_tuning

env_name = "pendulum"

default_config = {
    
    # Network parameters
    "value_sizes": (256,) * 5,
    "activation": "tanh",
    
    # Fixed parameters to choose
    "num_timesteps": 25_000_000,
    "num_evals": 20,
    "num_envs": 2048,
    "batch_size": 2048,
    "num_updates_per_batch": 8,
    
    # Hyperparameters to tune (default value here)
    "reward_scaling": 0.1,
    "unroll_length": 10,
    "discounting": 0.95,
    "learning_rate": 3e-4,
    
    # Lipschitz-bounded network params
    "gamma": 10.0,
    "train_lipschitz": False,
}

# Train MLP + LBDNs with many Lipschitz bounds over many iterations
# These sizes have been chosen so that the networks have a 
# similar number of trainable parameters
networks = ["mlp", "lbdn"]
sizes = [(32,) * 4, (21,) * 4]
tuning_config = {"gamma": [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]}
seeds = list(range(10))

for k in range(len(networks)):
    default_config["network"] = networks[k]
    default_config["policy_sizes"] = sizes[k]
    for seed in seeds:
        default_config["seed"] = seed
        train_hyperparam_tuning(env_name, default_config, tuning_config,
                                env_subdirpath="trained-models/")
