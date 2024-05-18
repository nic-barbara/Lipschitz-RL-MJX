# https://gymnasium.farama.org/environments/classic_control/pendulum/
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py

import jax
import jax.numpy as jnp
import mujoco

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx
from pathlib import Path
from .utils import scale_actions

class Pendulum(PipelineEnv):
    """Re-implementation of OpenAI Gymnasium Pendulum v1.
    
    This environment has the following modifications:
        - The pendulum velocity is not restricted.
        
    ### Action Space

    The agent take a 1-element vector for actions. The action space is a
    continuous `(action)` in `[-2, 2]`, where `action` represents the numerical
    torque applied to the pendulum (with magnitude representing the amount of torque and
    sign representing the direction)

    | Num | Action                         | Control Min | Control Max | Name (in corresponding config) | Joint | Unit        |
    |-----|--------------------------------|-------------|-------------|--------------------------------|-------|-------------|
    | 0   | torque applied on the pendulum | -1          | 1           | torque_servo                   | pin   | Torque (Nm) |
    
    NOTE: The input action is assumed to be within `[-1, 1]`. It is (linearly) 
    scaled up to `[-2, 2]` within the environment's `step()` call.
    """
    
    def __init__(
        self,
        max_steps=200,
        reset_pos_range=jnp.pi,
        reset_vel_range=1.0,
        angle_cost_weight=1,
        angvel_cost_weight=0.1,
        ctrl_cost_weight=0.001,
        **kwargs
    ):
        
        # Path to assets
        dirpath = Path(__file__).resolve().parent
        model_path = str(dirpath / "assets/pendulum.xml")
        
        # Read the model and set solver options
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        
        # Initialise from parent class
        sys = mjcf.load_model(mj_model)
        kwargs["backend"] = "mjx"
        super().__init__(sys, **kwargs)
        
        # Set attributes
        self._max_time = max_steps * self.sys.opt.timestep
        self._reset_pos_range = reset_pos_range
        self._reset_vel_range = reset_vel_range
        self._angle_cost_weight = angle_cost_weight
        self._angvel_cost_weight = angvel_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight        
    
    def reset(self, rng: jnp.ndarray) -> State:
        
        # Never re-use RNG in JAX
        rng1, rng2 = jax.random.split(rng, 2)
        
        # Set states and velocities
        qpos = jax.random.uniform(
            rng1, (self.sys.nq,),
            minval=-self._reset_pos_range,
            maxval=self._reset_pos_range)
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,),
            minval=-self._reset_vel_range,
            maxval=self._reset_vel_range)
        
        # Pack it all into a new state
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jnp.zeros(self.sys.nu))
        reward, done = jnp.zeros(2)
        metrics = {}
        
        return State(data, obs, reward, done, metrics)
        
    def step(self, state: State, action: jnp.ndarray) -> State:
        
        # Process the action
        # NOTE: assumes action in [-1, 1] and scales up!!
        action = scale_actions(self.sys, action)
        
        # Take a physics step with the pipeline, limit velocity
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        
        # Clip/process values
        qpos = angle_normalize(data.qpos)
        qvel = data.qvel
        
        # Compute reward and termination condition
        reward = -(self._angle_cost_weight * jnp.square(qpos[0]) +
                   self._angvel_cost_weight * jnp.square(qvel[0]) + 
                   self._ctrl_cost_weight * jnp.square(action[0]))
        
        done = jnp.float32(data.time > self._max_time)
            
        # Fill the state with new information
        obs = self._get_obs(data, action)
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(self, data: mjx.Data, action: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.concatenate([
            jnp.cos(data.qpos),
            jnp.sin(data.qpos),
            data.qvel
        ])
    
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi