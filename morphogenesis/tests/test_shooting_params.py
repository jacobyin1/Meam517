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

    print("\n--- 3. Constructing the New Robot ---")
    new_sys = p_shooter.update_sys(env.sys, shifts)

    print(f"Original Leg Lengths (approx): {env.sys.body_pos[1:, 2]}")
    print(f"Optimized Leg Lengths (approx): {new_sys.body_pos[1:, 2]}")

    # --- NEW: SAVE TO XML ---
    print("\n--- Saving Optimized XML ---")
    save_optimized_xml(
        xml_path=robot_xml_path,
        output_path=output_xml_path,
        model=env._mj_model,
        included_ids=p_shooter.included_ids,
        end_ids=p_shooter.end_ids,
        shifts=shifts
    )
    # ------------------------

    print("\n--- 4. Running Simulation on New Robot ---")
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

    print("\n--- 5. Rendering ---")
    # BONUS: Load the NEW optimized XML for the visualizer!
    # This fixes the "Ghost Bones" issue because the renderer will now
    # match the physics.

    with open(output_xml_path, 'r') as f:
        new_xml_string = f.read()

    # We create a temporary env just to get the new renderer model
    # (In a real app, you might just reload the visualizer separately)
    viz_env = load_environment(new_xml_string, config_path)
    viz = Visualizer(viz_env)

    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollout_states, output_filename, framerate=fps)

    print(f"Video saved to {os.path.abspath(output_filename)}")

    os.startfile(os.path.join(os.getcwd(), output_filename))
