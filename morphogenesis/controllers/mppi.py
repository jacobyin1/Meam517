from functools import partial

import jax
import jax.numpy as jnp


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

    def get_action(self, current_state, current_plan, rng):
        rng, subkey = jax.random.split(rng)
        actions = jax.random.normal(subkey, shape=(self.n_samples, self.n_steps, self.env.action_size)) * self.sigma
        actions = actions + current_plan
        actions = jnp.clip(actions, -1.0, 1.0)

        costs = jax.vmap(self._trajectory_cost, in_axes=(0, None))(actions, current_state)

        min_cost = jnp.min(costs)
        weights = jnp.exp(-(costs - min_cost) / self.temperature)
        weights = weights / jnp.sum(weights)

        new_plan = jnp.sum(weights[:, None, None] * actions, axis=0)
        return new_plan, rng

    def get_actions(self, init_state, rng):
        def step_fn(carry_state, _):
            current_state, current_plan, rng = carry_state
            jax.debug.print("ha")
            plan, new_rng = self.get_action(current_state, current_plan, rng)
            jax.debug.print("here")
            action = plan[0]
            new_plan = jnp.roll(plan, shift=-1, axis=0)
            new_plan = new_plan.at[-1].set(0.0)

            new_state = self.env.step(current_state, action)
            jax.debug.print("here1")
            return (new_state, new_plan, new_rng), (action, new_state)

        _, outs = jax.lax.scan(
            step_fn,
            (init_state, self.plan, rng),
            None,
            length=self.n_steps
        )

        jax.debug.print("{}", outs)
        actions, states = outs
        return actions, states

    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_cost(self, action_sequence, start_state):
        self.env.reset(jax.random.PRNGKey(0))
        def step_fn(current_state, action):
            next_state = self.env.step(current_state, action)
            cost = -next_state.reward
            return next_state, cost

        final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
        return jnp.sum(costs)