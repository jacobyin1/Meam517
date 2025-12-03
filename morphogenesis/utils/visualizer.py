import mujoco
import jax
import numpy as np
import imageio
from typing import List, Optional


class Visualizer:
    def __init__(self, env):
        """
        Args:
            env: Your ConcreteMJXCrawler instance.
        """
        # We need the CPU model for rendering.
        # BaseMJXCrawler stores this as self._mj_model
        if hasattr(env, 'unwrapped'):
            self.model = env.unwrapped._mj_model
        else:
            self.model = env._mj_model

        self.height = 480
        self.width = 640

        # Create a standard MuJoCo renderer
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        # Create a dummy standard data object to hold the state
        self.data = mujoco.MjData(self.model)

    def render_video(self, states: List, output_path: str = "rollout.mp4", framerate: int = 24,
                     camera_name: Optional[str] = None):
        """
        Takes a list of Brax States (from a JAX rollout), renders them, and saves MP4.
        """
        print(f"Rendering {len(states)} frames to {output_path}...")
        frames = []

        camera_arg = -1
        if camera_name is not None:
            camera_arg = camera_name

        for state in states:
            # 1. Extract qpos from the JAX state
            # We use state.pipeline_state.qpos because that holds the physics position
            jax_qpos = state.pipeline_state.qpos

            # 2. Transfer from GPU (JAX) to CPU (Numpy)
            # This is critical! Renderer cannot read GPU arrays directly.
            cpu_qpos = jax.device_get(jax_qpos)

            # 3. Update the CPU MuJoCo Data
            self.data.qpos[:] = cpu_qpos

            # We must forward the simulation kinematics to update geometry positions
            # based on the new qpos
            mujoco.mj_forward(self.model, self.data)

            # 4. Render
            self.renderer.update_scene(self.data, camera=camera_arg)
            pixels = self.renderer.render()
            frames.append(pixels)

        # 5. Save using imageio
        imageio.mimsave(output_path, frames, fps=framerate)
        print("Done.")

    def render_frame(self, state, camera_name: Optional[str] = None):
        """Returns a single numpy image (height, width, 3) for the given state."""
        # FIX: Handle camera logic here as well
        camera_arg = -1
        if camera_name is not None:
            camera_arg = camera_name
        jax_qpos = state.pipeline_state.qpos
        self.data.qpos[:] = jax.device_get(jax_qpos)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera_arg)
        return self.renderer.render()