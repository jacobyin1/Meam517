from functools import partial

import jax
import jax.numpy as jnp

def trajectory_cost(env_step_fn, action_sequence, start_state):
    def step_fn(current_state, action):
        next_state = env_step_fn(current_state, action)
        cost = -next_state.reward
        return next_state, cost

    final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
    return jnp.sum(costs)

def get_action(env_step_fn, current_state, current_plan, plan_noise, temperature):
    actions = current_plan + plan_noise
    actions = jnp.clip(actions, -1.0, 1.0)

    costs = jax.vmap(trajectory_cost, in_axes=(None, 0, None))(env_step_fn, actions, current_state)
    costs = jnp.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)

    min_cost = jnp.min(costs)
    exponent = -(costs - min_cost) / temperature
    exponent = jnp.clip(exponent, a_min=-500.0, a_max=0.0)
    weights = jnp.exp(exponent)

    weight_sum = jnp.sum(weights)
    weights = jnp.where(weight_sum < 1e-10,
                        jnp.ones_like(weights) / weights.shape[0],
                        weights / weight_sum)

    new_plan = jnp.sum(weights[:, None, None] * actions, axis=0)

    info = {
        "min_cost": min_cost,
        "mean_cost": jnp.mean(costs),
        "effective_samples": 1.0 / jnp.sum(weights**2),
        "weight_max": jnp.max(weights)
    }

    return new_plan, info

@partial(jax.jit, static_argnames=("env_step_fn", "n_steps", "horizon", "n_samples", "action_size"))
def _get_actions(env_step_fn,
                n_steps,
                horizon,
                n_samples,
                temperature,
                sigma,
                action_size,
                init_state,
                init_plan,
                rng):

    def step_fn(carry_state, step_index):
        current_state, current_plan, rng = carry_state

        rng, subkey = jax.random.split(rng)
        action_noise = jax.random.normal(subkey, shape=(n_samples, horizon, action_size)) * sigma

        plan, info = get_action(env_step_fn, current_state, current_plan, action_noise, temperature)

        action = plan[0]
        new_plan = jnp.roll(plan, shift=-1, axis=0)
        new_plan = new_plan.at[-1].set(0.0)

        new_state = env_step_fn(current_state, action)
        info = {
            **info,
            "height": new_state.pipeline_state.qpos[2],
            "velocity_x": new_state.pipeline_state.qvel[0],
            "velocity_y": new_state.pipeline_state.qvel[1],
            "action_magnitude": jnp.linalg.norm(action)
        }
        jax.debug.print("Step {}", step_index)

        return (new_state, new_plan, rng), (action, new_state, info)

    _, outs = jax.lax.scan(
        step_fn,
        (init_state, init_plan, rng),
        jnp.arange(n_steps),
        length=n_steps
    )

    actions, states, info = outs
    return actions, states, info


class Mppi:
    def __init__(self, env, n_steps=30, horizon=30, n_samples=1024, temperature=1.0, sigma=0.5, config=None):
        if config is not None:
            n_steps = config.get("n_steps", n_steps)
            horizon = config.get("horizon", horizon)
            n_samples = config.get("n_samples", n_samples)
            temperature = config.get("temperature", temperature)
            sigma = config.get("sigma", sigma)
        self.env = env
        self.n_steps = n_steps
        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        self.sigma = sigma
        self.plan = jnp.zeros((self.n_steps, env.action_size))

    def get_actions(self, rng):
        init_plan = jnp.zeros((self.horizon, self.env.action_size))

        actions, states, info = _get_actions(
            self.env.step,
            self.n_steps,
            self.horizon,
            self.n_samples,
            self.temperature,
            self.sigma,
            self.env.action_size,
            self.env.reset(rng),
            init_plan,
            rng,
        )

        self.plan = actions
        return actions, states.pipeline_state, info

    # def get_action(self, current_state, current_plan, rng):
    #     rng, subkey = jax.random.split(rng)
    #     actions = jax.random.normal(subkey, shape=(self.n_samples, self.n_steps, self.env.action_size)) * self.sigma
    #     actions = actions + current_plan
    #     actions = jnp.clip(actions, -1.0, 1.0)
    #
    #     costs = jax.vmap(self._trajectory_cost, in_axes=(0, None))(actions, current_state)
    #
    #     min_cost = jnp.min(costs)
    #     weights = jnp.exp(-(costs - min_cost) / self.temperature)
    #     weights = weights / jnp.sum(weights)
    #
    #     new_plan = jnp.sum(weights[:, None, None] * actions, axis=0)
    #     return new_plan, rng
    #
    # def get_actions(self, init_state, rng):
    #     def step_fn(carry_state, _):
    #         current_state, current_plan, rng = carry_state
    #         jax.debug.print("ha")
    #         plan, new_rng = self.get_action(current_state, current_plan, rng)
    #         jax.debug.print("here")
    #         action = plan[0]
    #         new_plan = jnp.roll(plan, shift=-1, axis=0)
    #         new_plan = new_plan.at[-1].set(0.0)
    #
    #         new_state = self.env.step(current_state, action)
    #         jax.debug.print("here1")
    #         return (new_state, new_plan, new_rng), (action, new_state)
    #
    #     _, outs = jax.lax.scan(
    #         step_fn,
    #         (init_state, self.plan, rng),
    #         None,
    #         length=self.n_steps
    #     )
    #
    #     jax.debug.print("{}", outs)
    #     actions, states = outs
    #     return actions, states
    #
    # @partial(jax.jit, static_argnums=(0,))
    # def _trajectory_cost(self, action_sequence, start_state):
    #     self.env.reset(jax.random.PRNGKey(0))
    #     def step_fn(current_state, action):
    #         next_state = self.env.step(current_state, action)
    #         cost = -next_state.reward
    #         return next_state, cost
    #
    #     final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
    #     return jnp.sum(costs)