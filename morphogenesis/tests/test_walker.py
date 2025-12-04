import json
import os
import jax
from morphogenesis.envs.env_loader import load_environment
from morphogenesis.controllers.mpc import MPC
from morphogenesis.utils.visualizer import Visualizer

CONFIG_PATH = "configs/train_normal.json"
OUTPUT_FILENAME = "tests/videos/walker_mpc_test.mp4"
ROBOT_PATH = "./tests/walker.xml"
with open(ROBOT_PATH, 'r') as f:
    robot_xml_string = f.read()

jax.config.update("jax_debug_nans", True)

def main():
    env = load_environment(robot_xml_string, CONFIG_PATH)
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # rng = jax.random.PRNGKey(0)
    # state = env.reset(rng)
    #
    # qpos = state.pipeline_state.qpos
    #
    # def single_step_cost(act):
    #     next_state = env.step(state, act)
    #     return -next_state.reward
    #
    # action = jnp.zeros(env.action_size)
    # grad_fn = jax.jacfwd(single_step_cost)
    # grads = grad_fn(action)
    # print(f"Gradients: {grads}")
    #


    mpc = MPC(
        env,
        horizon=config["horizon"],
        n_updates=config["n_updates"],
        lr=config["lr"]
    )

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    steps = config["n_steps"]
    rollout_states = [state]

    for i in range(steps):
        actions, rng = mpc.get_action(state, rng, n_actions=config["n_actions"])
        for j in range(config["n_actions"]):
            action = actions[j]
            state = env.step(state, action)
            rollout_states.append(state)
            z_height = state.pipeline_state.qpos[2]
            print(f"Step {i}: Reward={state.reward:.3f}, Z-Height={z_height:.3f}")

    viz = Visualizer(env)
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    viz.render_video(rollout_states, OUTPUT_FILENAME, framerate=fps)

    os.startfile(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()