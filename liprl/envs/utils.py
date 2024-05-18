import jax.numpy as jnp
from brax.base import System

def scale_actions(sys: System, action: jnp.ndarray) -> jnp.ndarray:
    """
    Scale an input action from `[-1, 1]` to control limits
    of each actuator in an MJX model.
    
    We assume the action is in `[-1, 1]` and apply a linear transform
    to scale the control to `[a, b]` with `u = (u + 1)(b-a)/2 + a`
    """
    mins = sys.actuator_ctrlrange[:, 0]
    maxs = sys.actuator_ctrlrange[:, 1]
    return (action + 1) * (maxs - mins) * 0.5 + mins
