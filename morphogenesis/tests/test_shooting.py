import json
import os
import jax
import jax.numpy as jnp

from morphogenesis.controllers.shooting import Shooting
from morphogenesis.envs.env_loader import load_environment
from morphogenesis.utils.save_info import save_log
from morphogenesis.utils.visualizer import Visualizer

CONFIG_PATH = "configs/train_lbfgs_shooting.json"
OUTPUT_FILE_VIDEO = "tests/videos/walker_shooting_lbfgs_test.mp4"
OUTPUT_FILE_METRICS = "tests/metrics/walker_shooting_lbfgs_metrics.json"
ROBOT_PATH = "./tests/walker.xml"
with open(ROBOT_PATH, 'r') as f:
    robot_xml_string = f.read()

jax.config.update("jax_debug_nans", True)

def main():
    env = load_environment(robot_xml_string, CONFIG_PATH)
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    shooting = Shooting(env, config=config)
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    actions, info, rng = shooting.get_action(state, rng)

    def step_fn(carry_state, action):
            next_state = env.step(carry_state, action)
            new_info = {
                **info,
                "height": next_state.pipeline_state.qpos[2],
                "velocity_x": next_state.pipeline_state.qvel[0],
                "velocity_y": next_state.pipeline_state.qvel[1],
                "action_magnitude": jnp.linalg.norm(action)
            }
            return next_state, (next_state, new_info)

    _, results = jax.lax.scan(
        step_fn,
        state,
        actions
    )

    rollouts, info = results
    save_log(OUTPUT_FILE_METRICS, actions, rollouts, info)

    viz = Visualizer(env)
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollouts, OUTPUT_FILE_VIDEO, framerate=fps)

    os.startfile(os.path.join(os.getcwd(), OUTPUT_FILE_VIDEO))


if __name__ == "__main__":
    main()