import json
import os
import jax

from morphogenesis.controllers.mppi import Mppi
from morphogenesis.envs.env_loader import load_environment
from morphogenesis.utils.save_info import save_log
from morphogenesis.utils.visualizer import Visualizer

CONFIG_PATH = "configs/train_normal_mppi.json"
OUTPUT_FILE_VIDEO = "tests/videos/walker_mppi_test.mp4"
OUTPUT_FILE_METRICS = "tests/metrics/walker_mppi_metrics.json"
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

    actions, rollouts, info = mppi.get_actions(rng)
    save_log(OUTPUT_FILE_METRICS, actions, rollouts, info)

    viz = Visualizer(env)
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollouts, OUTPUT_FILE_VIDEO, framerate=fps)

    os.startfile(os.path.join(os.getcwd(), OUTPUT_FILE_VIDEO))


if __name__ == "__main__":
    main()