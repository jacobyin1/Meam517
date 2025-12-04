import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from morphogenesis.utils.xml_merger import merge_robot_and_env
from brax.envs.base import Env, ObservationSize, State


class BaseMJXEnv(Env):
    """
    Brax class for env wrapping.
    """

    def __init__(self, xml_path, robot_xml_string, params, n_frames: int = 1):
        self.xml_path = xml_path
        self.params = params

        combined_xml_string = merge_robot_and_env(robot_xml_string, xml_path)
        self._mj_model = mujoco.MjModel.from_xml_string(combined_xml_string) # noqa
        self.sys = mjx.put_model(self._mj_model)

        self._backend = 'mjx'
        self.n_frames = n_frames

    def reset(self, rng):
        reward = jnp.zeros(())
        done = jnp.zeros(())

        qvel = jnp.zeros(self.sys.nv)
        start_speed = 0.5
        qvel = qvel.at[0].set(start_speed)
        qvel = qvel.at[1].set(start_speed)

        data = mjx.make_data(self.sys)
        data = data.replace(qpos=self.sys.qpos0, qvel=qvel)
        metrics = {}

        return State(
            pipeline_state=data, # noqa
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def step(self, state, action):
        data = self.pipeline_step(state.pipeline_state, action)
        reward = self.compute_reward(data, action)
        done = self.check_termination(data)
        return state.replace( # noqa
            pipeline_state=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
        )

    def pipeline_step(self, data, action):
        def f(data, _):
            ctrl = action
            data = data.replace(ctrl=ctrl)
            return mjx.step(self.sys, data), None

        data, _ = jax.lax.scan(f, data, (), self.n_frames)
        return data

    def _get_obs(self, data):
        return jnp.concatenate([data.qpos[7:], data.qvel])

    def check_termination(self, data):
        raise NotImplementedError("Subclass must implement check_termination")

    def compute_reward(self, state, action):
        raise NotImplementedError("Subclass must implement compute_reward")

    @property
    def observation_size(self) -> ObservationSize:
        return self.sys.nq - 7 + self.sys.nv

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def backend(self) -> str:
        return self._backend

