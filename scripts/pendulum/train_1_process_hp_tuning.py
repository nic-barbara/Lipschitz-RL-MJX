from liprl.tuning import HyperparamTuningResults

# Testing below
tuning_params = ["reward_scaling",
                 "unroll_length",
                 "discounting",
                 "learning_rate", 
                 "gamma"]
h = HyperparamTuningResults("pendulum", tuning_params)
h.plot_rewards()
