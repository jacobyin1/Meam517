import optax
import jax
import jax.numpy as jnp


class MPC:
    def __init__(self, env, horizon: int = 10, n_updates: int = 20, lr: float = 0.1):

        self.env = env
        self.horizon = horizon
        self.n_updates = n_updates
        self.action_size = env.action_size

        self.plan = jnp.zeros((self.horizon, self.action_size))

        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=n_updates,
            alpha=0.1
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule)
        )

    def get_action(self, current_state, rng, n_actions=1):
        opt_state = self.optimizer.init(self.plan)

        def optimization_step(carry, step_index):
            current_plan, current_opt_state = carry
            grads = jax.jacfwd(self._trajectory_cost, argnums=0)(current_plan, current_state)

            jax.debug.print("step {}, max grad {}, cost {}",
                            step_index,
                            jnp.max(jnp.abs(grads)),
                            self._trajectory_cost(current_plan, current_state))

            updates, new_opt_state = self.optimizer.update(grads, current_opt_state, current_plan)
            new_plan = optax.apply_updates(current_plan, updates)
            new_plan = jnp.clip(new_plan, -1.0, 1.0)

            return (new_plan, new_opt_state), None

        xs = jnp.arange(self.n_updates)

        (self.plan, _), _ = jax.lax.scan(
            optimization_step,
            (self.plan, opt_state),
            xs,
            length=self.n_updates
        )

        actions_to_execute = self.plan[:n_actions]
        self.plan = jnp.roll(self.plan, shift=-n_actions, axis=0)
        self.plan = self.plan.at[-n_actions:].set(0.0)

        if n_actions == 1:
            return actions_to_execute[0], rng
        else:
            return actions_to_execute, rng


    def _trajectory_cost(self, action_sequence, start_state):

        def step_fn(current_state, action):
            next_state = self.env.step(current_state, action)
            cost = -next_state.reward
            return next_state, cost

        final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
        return jnp.sum(costs)