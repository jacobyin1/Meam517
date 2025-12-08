import mujoco
import jax
import numpy as np
import imageio
from typing import List, Optional


class Visualizer:
    def __init__(self, env):

        if hasattr(env, 'unwrapped'):
            self.model = env.unwrapped._mj_model
        else:
            self.model = env._mj_model

        self.height = 480
        self.width = 640

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.data = mujoco.MjData(self.model)

    def render_video(self, states, output_path: str = "rollout.mp4", framerate: int = 24,
                     camera_name: Optional[str] = None):

        print(f"Rendering {states.qpos.shape[0]} frames to {output_path}...")
        frames = []

        camera_arg = -1
        if camera_name is not None:
            camera_arg = camera_name

        for jax_qpos in states.qpos:
            cpu_qpos = jax.device_get(jax_qpos)

            self.data.qpos[:] = cpu_qpos
            mujoco.mj_forward(self.model, self.data)

            self.renderer.update_scene(self.data, camera=camera_arg)
            pixels = self.renderer.render()
            frames.append(pixels)

        imageio.mimsave(output_path, frames, fps=framerate)
        print("Done.")

    def render_frame(self, state, camera_name: Optional[str] = None):
        camera_arg = -1
        if camera_name is not None:
            camera_arg = camera_name
        jax_qpos = state.pipeline_state.qpos
        self.data.qpos[:] = jax.device_get(jax_qpos)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera_arg)
        return self.renderer.render()