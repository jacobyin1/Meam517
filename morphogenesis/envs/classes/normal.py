import jax.numpy as jnp
from .base import BaseMJXEnv


class NormalEnv(BaseMJXEnv):
    def compute_reward(self, state, action):
        vel_reward = 5.0 * state.qd[0]
        ctrl_cost = -0.1 * jnp.sum(action ** 2)
        return vel_reward + ctrl_cost

    def check_termination(self, state):
        """
        No Termination Condition.
        """
        return jnp.zeros(())