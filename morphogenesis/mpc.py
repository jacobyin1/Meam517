import optax
import jax
import jax.numpy as jnp


class MPC:
    def __init__(self, env, horizon: int = 10, n_updates: int = 20, learning_rate: float = 0.1):

        self.env = env
        self.horizon = horizon
        self.n_updates = n_updates
        self.action_size = env.action_size

        self.plan = jnp.zeros((self.horizon, self.action_size))
        self.optimizer = optax.adam(learning_rate=learning_rate)

    def get_action(self, current_state, rng):
        opt_state = self.optimizer.init(self.plan)

        def optimization_step(carry, _):
            current_plan, current_opt_state = carry
            grads = jax.jacfwd(self._trajectory_cost, argnums=0)(current_plan, current_state)
            updates, new_opt_state = self.optimizer.update(grads, current_opt_state, current_plan)
            new_plan = optax.apply_updates(current_plan, updates)
            new_plan = jnp.clip(new_plan, -1.0, 1.0)

            return (new_plan, new_opt_state), None

        (self.plan, _), _ = jax.lax.scan(
            optimization_step,
            (self.plan, opt_state),
            None,
            length=self.n_updates
        )

        action_to_execute = self.plan[0]
        self.plan = jnp.roll(self.plan, shift=-1, axis=0)
        self.plan = self.plan.at[-1].set(jnp.zeros(self.action_size))

        return action_to_execute, rng

    def _trajectory_cost(self, action_sequence, start_state):

        def step_fn(current_state, action):
            next_state = self.env.step(current_state, action)
            cost = -next_state.reward
            return next_state, cost

        final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
        return jnp.sum(costs)