import json
import os
import jax

from morphogenesis.controllers.mppi import Mppi
from morphogenesis.envs.env_loader import load_environment
from morphogenesis.utils.visualizer import Visualizer

CONFIG_PATH = "configs/train_normal_mppi.json"
OUTPUT_FILENAME = "tests/videos/walker_mppi_test.mp4"
ROBOT_PATH = "./tests/walker.xml"
with open(ROBOT_PATH, 'r') as f:
    robot_xml_string = f.read()

jax.config.update("jax_debug_nans", True)

def main():
    env = load_environment(robot_xml_string, CONFIG_PATH)
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    mppi = Mppi(env, config=config)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    actions, rollout_states = mppi.get_actions(state, rng)

    # rollout_states = rollout_states_pipeline.pipeline_state

    viz = Visualizer(env)
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollout_states, OUTPUT_FILENAME, framerate=fps)

    os.startfile(os.path.join(os.getcwd(), OUTPUT_FILENAME))


if __name__ == "__main__":
    main()