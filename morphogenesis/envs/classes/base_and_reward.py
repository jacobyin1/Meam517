from morphogenesis.envs.classes.base import BaseMJXEnv
from mujoco import mjx
import jax.numpy as jnp
import jax
from typing import Callable, Any, List


class RewardMJXEnv(BaseMJXEnv):

    def __init__(self,
                 xml_string: str,
                 robot_xml_string: str,
                 params: Any,
                 reward_fns: List[Callable],
                 reward_weights: jax.Array,
                 done_fn: Callable,
                 n_frames: int = 10):
        super().__init__(xml_string, robot_xml_string, params, n_frames=n_frames)
        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        self.done_fn = done_fn

    def compute_reward(self, data: mjx.Data, action: jax.Array):
        rew = jnp.zeros(())
        for i in range(len(self.reward_fns)):
            rew += self.reward_weights[i] * self.reward_fns[i](data, action)
        return rew

    def check_termination(self, data):
        return self.done_fn(data)