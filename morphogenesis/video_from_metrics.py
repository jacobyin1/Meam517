import mujoco
import numpy as np
import imageio
import json
import os
from typing import List, Optional

from morphogenesis.envs.env_loader import load_environment

CONFIG_PATH = "configs/train_normal_mppi.json"
OUTPUT_FILE_VIDEO = "tests/videos/walker_mppi_test.mp4"
OUTPUT_FILE_METRICS = "tests/metrics/walker_mppi_metrics.json"
ROBOT_PATH = "tests/walker.xml"

class Visualizer:
    def __init__(self, env):
        # We need the model from the environment to know what geometry to render
        if hasattr(env, 'unwrapped'):
            self.model = env.unwrapped._mj_model
        else:
            self.model = env._mj_model

        self.height = 480
        self.width = 640

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.data = mujoco.MjData(self.model)

    def render_from_json(self, json_path: str, output_path: str = "json_rollout.mp4", framerate: int = 24,
                         camera_name: Optional[str] = None):
        """
        Reads a JSON file containing trajectory data and renders it to a video.
        Expected JSON structure: { "trajectory": { "qpos":List[List[float]], ... } }
        """

        if not os.path.exists(json_path):
            print(f"Error: The file '{json_path}' was not found.")
            return

        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 1. Extract qpos history
        try:
            qpos_history = data["trajectory"]["qpos"]
        except KeyError:
            print("Error: Could not find ['trajectory']['qpos'] in the JSON file.")
            return

        print(f"Rendering {len(qpos_history)} frames to {output_path}...")
        frames = []

        camera_arg = -1
        if camera_name is not None:
            camera_arg = camera_name

        # 2. Iterate through the history
        for frame_idx, qpos in enumerate(qpos_history):
            # qpos is a standard Python list of floats here

            # 3. Update MuJoCo Internal Data
            # We copy the positions from the JSON into the simulation state
            self.data.qpos[:] = np.array(qpos)

            # 4. Forward Kinematics
            # This updates the locations of all bodies based on the new qpos
            mujoco.mj_forward(self.model, self.data)

            # 5. Render
            self.renderer.update_scene(self.data, camera=camera_arg)
            pixels = self.renderer.render()
            frames.append(pixels)

        # 6. Save Video
        imageio.mimsave(output_path, frames, fps=framerate)
        print("Done.")


# --- Usage Example ---
if __name__ == "__main__":
    with open(ROBOT_PATH, 'r') as f:
        robot_xml_string = f.read()
    env = load_environment(robot_xml_string, CONFIG_PATH)
    viz = Visualizer(env)
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_from_json("tests/metrics/walker_mppi_metrics.json", OUTPUT_FILE_VIDEO, framerate=fps)

    os.startfile(os.path.join(os.getcwd(), OUTPUT_FILE_VIDEO))

