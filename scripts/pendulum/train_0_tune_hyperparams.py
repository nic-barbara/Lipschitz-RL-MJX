from liprl.tuning import train_hyperparam_tuning

env_name = "pendulum"

default_config = {
    
    # Network parameters
    "seed": 0,
    "value_sizes": (256,) * 5, # (Default in Brax)
    "activation": "tanh",
    
    # Fixed parameters to choose
    "num_timesteps": 30_000_000,
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

tuning_config = {
    "reward_scaling": [0.01, 0.1, 1.0],
    "unroll_length": [10, 20, 40],
    "discounting": [0.99, 0.97, 0.95, 0.93],
    "learning_rate": [1e-4, 3e-4, 5e-4, 7e-4],
    "gamma": [1.0, 10.0, 20.0],
}

# Train different networks/sizes
# These sizes have been chosen so that the networks have a 
# similar number of trainable parameters
networks = ["mlp", "lbdn"]
sizes = [(32,) * 4, (21,) * 4]

for k in range(len(networks)):
    default_config["network"] = networks[k]
    default_config["policy_sizes"] = sizes[k]
    train_hyperparam_tuning(env_name, default_config, tuning_config, 
                            env_subdirpath="tuning/")
