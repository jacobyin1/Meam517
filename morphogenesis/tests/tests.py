import os

import jax
import jax.numpy as jnp
from morphogenesis.envs.env_loader import load_environment
from morphogenesis.utils.visualizer import Visualizer

# Define paths (assuming the script runs from the project root)
CONFIG_PATH = "configs/train_normal.json"
with open("./tests/test.xml", 'r') as f:
    xml_string = f.read()


def test_environment_functionality():
    """
    Tests initialization, stepping, and differentiability.
    """
    print("--- 1. Testing Environment Initialization ---")

    # 1. Load the customized environment
    env = load_environment(xml_string, CONFIG_PATH)

    print(env.action_size)
    assert env.action_size == 1, "Action size mismatch for placeholder robot."

    print("\n--- 2. Testing Reset and Step Functionality (Rollout) ---")

    # JIT compile the functions for speed
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)

    # Initial Reset
    state = jit_reset(key_reset)
    print(f"Reset successful. Initial Reward: {state.reward}")

    # Perform a few steps (applying a constant positive torque)
    action = jnp.array([1.0])  # Apply maximum positive torque to the hip motor

    for i in range(5):
        rng, key_step = jax.random.split(rng)
        state = jit_step(state, action)

        # Check termination condition
        if state.done:
            print(f"Episode finished at step {i + 1}.")
            break

        print(f"Step {i + 1}: Reward={state.reward:.4f}, Z-Height={state.pipeline_state.qpos[2]:.4f}")

    print("SUCCESS: Environment steps and computes rewards/done flags.")

    print("\n--- 3. Testing Differentiability (MPC Requirement) ---")

    # -------------------------------------------------------------
    # CRITICAL: Use 'jax.lax.scan' for the trajectory loop
    # -------------------------------------------------------------
    def trajectory_cost(start_state, actions):
        """Calculates cost using jax.lax.scan for fast compilation."""

        def step_fn(current_state, action):
            next_state = env.step(current_state, action)
            # MPC minimizes cost, which is -reward
            cost = -next_state.reward
            return next_state, cost

        # scan over the actions
        final_state, costs = jax.lax.scan(step_fn, start_state, actions)
        return jnp.sum(costs)

    # Short horizon for testing
    horizon = 5
    initial_actions = jnp.ones((horizon, env.action_size)) * 0.1

    print("Calculating gradients via Forward-Mode AD (jacfwd)...")

    # -------------------------------------------------------------
    # THE FIX: Use jax.jacfwd instead of jax.grad
    # Forward-mode differentiation handles 'while' loops (inside MJX solver)
    # much better than Reverse-mode.
    # -------------------------------------------------------------
    grad_fn = jax.jacfwd(trajectory_cost, argnums=1)

    grads = grad_fn(state, initial_actions)
    print(grads)
    print(f"Gradient calculated successfully.")
    print(f"Shape of gradients (Horizon, Actions): {grads.shape}")
    print(f"Sample Gradient Value: {grads[0]}")

    if not jnp.isnan(grads).any():
        print("SUCCESS: Gradients are non-NaN. The system is ready for Differentiable MPC!")
    else:
        print("FAILED: Gradients contain NaN values.")

def test_visualize():
    SIMULATION_STEPS = 200
    OUTPUT_FILENAME = "./tests/test.mp4"

    print(f"--- Loading Environment from {CONFIG_PATH} ---")

    # Load the environment using your Factory
    try:
        env = load_environment(xml_string, CONFIG_PATH)
    except Exception as e:
        print(f"Error loading environment: {e}")
        print("Make sure 'configs/test_config.json' exists.")
        return


    # 2. PREPARE JAX FUNCTIONS
    # We JIT compile reset and step for performance,
    # even though rendering is the bottleneck here.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 3. RUN SIMULATION (ROLLOUT)
    print(f"--- Simulating {SIMULATION_STEPS} steps ---")

    rng = jax.random.PRNGKey(42)  # Seed for reproducibility
    state = jit_reset(rng)

    rollout_states = []

    for i in range(SIMULATION_STEPS):
        # Generate a simple Sine wave action to make the robot "wiggle"
        # This proves the motors are working.
        # env.action_size ensures we send the right number of commands.
        action = jnp.ones(env.action_size) * jnp.sin(i * 0.1)

        # Step the physics
        state = jit_step(state, action)

        # Store state for rendering
        rollout_states.append(state)

        if i % 50 == 0:
            print(f"Simulated step {i}/{SIMULATION_STEPS}")

    # 4. RENDER VIDEO
    print("--- Rendering Video ---")
    viz = Visualizer(env)

    # Calculate correct FPS so video plays at real-time speed
    # framerate = 1 / (physics_timestep * action_repeat)
    # e.g., 1 / (0.005 * 5) = 40 FPS
    fps = int(1.0 / (env.sys.opt.timestep * env.n_frames))
    print(f"Rendering at {fps} FPS...")

    # Render!
    # You can pass camera_name="track" if you defined a camera in XML
    viz.render_video(rollout_states, OUTPUT_FILENAME, framerate=fps)

    print(f"SUCCESS: Video saved to {os.path.abspath(OUTPUT_FILENAME)}")

if __name__ == "__main__":
    test_environment_functionality()