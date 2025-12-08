import json
import jax
import os
from morphogenesis.utils.xml_saver import save_optimized_xml
import mujoco
import mujoco.mjx as mjx

from morphogenesis.tests.test_shooting import OUTPUT_FILENAME
from morphogenesis.utils.visualizer import Visualizer
from morphogenesis.controllers.shooting_params import ShootingParams
from morphogenesis.envs.env_loader import load_environment

class DummyState:
    def __init__(self, data):
        self.pipeline_state = data

def main():
    robot_xml_path = "./tests/walker.xml"
    output_xml_path = "./tests/new_walker.xml"
    with open("./tests/walker.xml", 'r') as f:
        robot_xml_string = f.read()
    config_path = "configs/train_param_shooting.json"
    output_filename = "tests/videos/walker_param_shooting_test.mp4"
    env = load_environment(robot_xml_string, config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    p_shooter = ShootingParams(env, config=config)

    shifts, actions = p_shooter.optimize()

    new_sys = p_shooter.update_sys(env.sys, shifts)

    save_optimized_xml(
        xml_path=robot_xml_path,
        output_path=output_xml_path,
        model=env._mj_model,
        included_ids=p_shooter.included_ids,
        end_ids=p_shooter.end_ids,
        shifts=shifts
    )

    data = mjx.make_data(new_sys)
    rollout_states = []

    for i in range(config["n_steps"]):
        action = actions[i]

        def physics_step(d, _):
            d = d.replace(ctrl=action)
            return mjx.step(new_sys, d), None

        data, _ = jax.lax.scan(physics_step, data, None, length=env.n_frames)
        rollout_states.append(DummyState(data))

        z_height = data.qpos[2]
        vx = data.qvel[0]
        print(f"Step {i}: Z-Height={z_height:.3f}, Velocity={vx:.3f}")

    with open(output_xml_path, 'r') as f:
        new_xml_string = f.read()

    viz_env = load_environment(new_xml_string, config_path)
    viz = Visualizer(viz_env)

    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollout_states, output_filename, framerate=fps)

    os.startfile(os.path.join(os.getcwd(), output_filename))
