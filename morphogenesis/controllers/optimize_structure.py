import jax
import jax.numpy as jnp
import jaxopt
import mujoco
import mujoco.mjx
from morphogenesis.envs.env_loader import load_environment

# CONFIG
CONFIG_PATH = "configs/test_config.json"
ROLLOUT_LEN = 40  # How far into the future to optimize


def main():
    # 1. Load Base Assets
    print("Loading Environment...")
    base_env = load_environment(CONFIG_PATH)
    sys = base_env.sys
    n_frames = base_env.n_frames
    action_size = base_env.action_size

    # 2. Identify Body Indices (for modifying leg lengths)
    names = [b.name for b in base_env._mj_model.body]
    idx_leg_r = names.index("leg_right")
    idx_foot_r = names.index("foot_right")
    idx_leg_l = names.index("leg_left")
    idx_foot_l = names.index("foot_left")

    # 3. The Morphing Function
    # Takes the base physics model and design params, returns a NEW model
    def update_model(base_sys, design_params):
        thigh_len, shin_len = design_params
        new_pos = base_sys.body_pos

        # Modify geometry (Negative Z relative to parent)
        new_pos = new_pos.at[idx_leg_r, 2].set(-thigh_len)
        new_pos = new_pos.at[idx_foot_r, 2].set(-shin_len)
        new_pos = new_pos.at[idx_leg_l, 2].set(-thigh_len)
        new_pos = new_pos.at[idx_foot_l, 2].set(-shin_len)

        return base_sys.tree_replace({'body_pos': new_pos})

    # 4. The Joint Cost Function
    # This is the function L-BFGS will minimize
    def joint_cost(params):
        # Unpack
        design = params['design']
        controls = params['controls']  # Shape (ROLLOUT_LEN, action_size)

        # A. Build the Robot
        # Gradients flow backwards from here to 'design'
        optimized_sys = update_model(sys, design)

        # B. Initialize Physics
        rng = jax.random.PRNGKey(0)
        qpos = optimized_sys.qpos0
        qvel = jnp.zeros(optimized_sys.nv)
        data = jax.jit(mujoco.mjx.make_data)(optimized_sys)

        # C. Simulation Loop
        def step_fn(carry, action):
            d, _ = carry

            # Manual physics sub-stepping
            def physics_loop(d_inner, _):
                d_inner = d_inner.replace(ctrl=action)
                return mujoco.mjx.step(optimized_sys, d_inner), None

            d_next, _ = jax.lax.scan(physics_loop, d, None, length=n_frames)

            # Reward Calculation
            vel_x = d_next.qvel[0]
            z_height = d_next.qpos[2]

            # Costs (Negative Reward)
            fall_cost = jnp.where(z_height < 0.25, 5.0, 0.0)  # Penalty for falling
            ctrl_cost = 0.001 * jnp.sum(action ** 2)  # Penalty for energy

            # We want to MAX velocity, so we MINIMIZE negative velocity
            step_cost = -vel_x + fall_cost + ctrl_cost

            return (d_next, None), step_cost

        # Run the rollout using the 'controls' from params
        _, step_costs = jax.lax.scan(step_fn, (data, rng), controls)

        return jnp.sum(step_costs)

    # 5. Forward-Mode Wrapper (CRITICAL)
    # This forces JAX to use jacfwd (efficient for physics) instead of grad.
    def val_and_grad_fwd(params):
        val = joint_cost(params)
        grad = jax.jacfwd(joint_cost)(params)
        return val, grad

    # 6. Setup L-BFGS-B
    print("Setting up L-BFGS-B Solver...")

    # Initialize Guesses
    init_design = jnp.array([0.45, 0.5])  # Thigh, Shin
    key = jax.random.PRNGKey(42)
    # Start with small random actions
    init_controls = jax.random.uniform(key, (ROLLOUT_LEN, action_size), minval=-0.1, maxval=0.1)

    init_params = {
        "design": init_design,
        "controls": init_controls
    }

    # Define Bounds (Constraints)
    # We must match the structure of init_params
    lower_bounds = {
        "design": jnp.array([0.2, 0.2]),  # Legs can't be shorter than 20cm
        "controls": jnp.full((ROLLOUT_LEN, action_size), -1.0)  # Motor Min
    }
    upper_bounds = {
        "design": jnp.array([0.8, 0.8]),  # Legs can't be longer than 80cm
        "controls": jnp.full((ROLLOUT_LEN, action_size), 1.0)  # Motor Max
    }

    # Initialize Solver
    solver = jaxopt.LBFGSB(
        fun=val_and_grad_fwd,  # Our custom forward-mode wrapper
        value_and_grad=True,  # Tell solver we provide (val, grad)
        maxiter=50,
        tol=1e-3,
        implicit_diff=False  # Must be False for physics loops
    )

    # 7. Run Optimization
    print("Running Optimization (this compiles first, please wait)...")

    # L-BFGS-B will now juggle leg lengths AND motor actions simultaneously
    # to find the fastest possible robot.
    sol = solver.run(
        init_params=init_params,
        bounds=(lower_bounds, upper_bounds)
    )

    # 8. Results
    final_params = sol.params
    final_loss = sol.state.value

    print("\n=== CO-DESIGN COMPLETE ===")
    print(f"Final Cost: {final_loss:.4f}")
    print(f"Optimal Thigh Length: {final_params['design'][0]:.4f} m")
    print(f"Optimal Shin Length:  {final_params['design'][1]:.4f} m")

    # Optional: Save results
    # jnp.save("optimized_params.npy", final_params)


if __name__ == "__main__":
    main()