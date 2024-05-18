# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPO policy/value network constructors.

Edited by: Nicholas Barbara
Email: nicholas.barbara@sydney.edu.au
"""

import dataclasses
import jax.numpy as jnp

from brax.training import types
from flax import linen as nn
from typing import Any, Callable, Sequence

from liprl.networks.lbdn import LBDN
from liprl.networks.mlp import MLP
from liprl.networks.utils import ActivationFn


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


def make_policy_network(
    obs_size: int,
    param_size: int,
    network: str = "mlp",
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    gamma: jnp.float32 = 1.0,
    trainable_lipschitz: bool = False,
    preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
) -> FeedForwardNetwork:
    """Create a Lipschitz-bounded policy network for RL.
    
    Choose between the following network architectures:
    
        - "mlp" (default)
        - "lbdn"
        
    The `gamma` and `trainable_lipschitz` arguments are only valid for networks
    with user-defined Lipschitz bounds.
    
    Arguments:
        obs_size: Input size (size of environment observations).
        param_size: Output size (size of output distribution vector).
        network: The type of network architecture to use (default: "mlp").
        hidden_layer_sizes: Hidden layer sizes (default: (256, 256)).
        activation: Activation function to use (default: relu).
        gamma: Upper bound on the Lipschitz constant (default: inf).
        trainable_lipschitz: Make the Lipschitz constant trainable (default: False).
        preprocess_observations_fn: Function to pre-process observations (default: identity).
    
    See https://proceedings.mlr.press/v202/wang23v.html information on the
    LBDN models.
    """
  
    layer_sizes = list(hidden_layer_sizes) + [param_size]
    
    if network == "mlp":
        policy_module = MLP(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=nn.initializers.lecun_uniform())
    elif network == "lbdn":
        policy_module = LBDN(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=nn.initializers.glorot_normal(),
            gamma=gamma,
            trainable_lipschitz=trainable_lipschitz)
    else:
        raise NotImplementedError("Network type {} is not implemented.".format(network))

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), 
        apply=apply)


def make_value_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
) -> FeedForwardNetwork:
    """Creates a value network for RL.

    This network approximates the value function for a policy. Unlike
    the policy network, this network does not have a restriction on
    the Lipschitz bound.
    
    Arguments:
        obs_size: Input size (size of environment observations).
        hidden_layer_sizes: Hidden layer sizes (default: (256, 256)).
        activation: Activation function to use (default: relu).
        preprocess_observations_fn: Function to pre-process observations (default: identity).
    """
  
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=nn.initializers.lecun_uniform())

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), 
        apply=apply)
