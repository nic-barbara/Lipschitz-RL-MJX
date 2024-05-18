import brax.envs
from liprl.envs import pendulum

brax.envs.register_environment('pendulum', pendulum.Pendulum)