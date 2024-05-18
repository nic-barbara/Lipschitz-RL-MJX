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
PPO MLP networks.

Edited by: Nicholas Barbara
Email: nicholas.barbara@sydney.edu.au
"""

import flax

from brax.training import distribution
from brax.training import types
from brax.training.types import PRNGKey

from flax import linen as nn
from jax import numpy as jnp
from typing import Sequence, Tuple

from liprl.networks.utils import ActivationFn
from liprl.networks import feedforward as networks


@flax.struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    network: str = "mlp",
    activation: ActivationFn = nn.tanh,
    gamma: jnp.float32 = 1.0,
    trainable_lipschitz: bool = False,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
) -> PPONetworks:
    """Make PPO networks with preprocessor, with an optional Lipschitz bound
    on the policy network.
    
    Choose between the following network architectures:
    
        - "mlp" (default)
        - "lbdn"
        
    The `gamma` and `trainable_lipschitz` arguments are only valid for networks
    with user-defined Lipschitz bounds.
    
    Arguments:
        observation_size: Input size (size of environment observations).
        action_size: Output size (size of environment actions).
        network: The type of network architecture to use (default: "mlp").
        activation: Activation function to use (default: tanh).
        gamma: Upper bound on the Lipschitz constant (default: inf).
        trainable_lipschitz: Make the Lipschitz constant trainable (default: False).
        policy_hidden_layer_sizes: Hidden layer sizes for policy (default: (32,) * 4).
        value_hidden_layer_sizes: Hidden layer sizes for policy (default: (256,) * 5).
        preprocess_observation_fn: Function to pre-process observations (default: identity).
        
    See also `make_policy_network`.
    """
    
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size)
    
    policy_network = networks.make_policy_network(
        observation_size,
        parametric_action_distribution.param_size,
        network=network,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        gamma=gamma,
        trainable_lipschitz=trainable_lipschitz,
        preprocess_observations_fn=preprocess_observations_fn)
    
    value_network = networks.make_value_network(
        observation_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn)

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


def make_inference_fn(ppo_networks: PPONetworks):
    """Create a function which constructs the policy given params."""

    def make_policy(params: types.PolicyParams,
                    deterministic: bool = False) -> types.Policy:
      
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: types.Observation,
                  key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
          
          logits = policy_network.apply(*params, observations)
          
          if deterministic:
              return ppo_networks.parametric_action_distribution.mode(logits), {}
            
          raw_actions = parametric_action_distribution.sample_no_postprocessing(
              logits, key_sample)
          
          log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
          
          postprocessed_actions = parametric_action_distribution.postprocess(
              raw_actions)
          
          return postprocessed_actions, {
              'log_prob': log_prob,
              'raw_action': raw_actions
          }

        return policy

    return make_policy
