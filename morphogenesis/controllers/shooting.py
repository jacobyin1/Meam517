import optax
import jax
import jaxopt
import jax.numpy as jnp
from functools import partial


class Shooting:
    def __init__(self, env, n_steps: int = 10, n_updates: int = 20):

        self.env = env
        self.n_steps = n_steps
        self.n_updates = n_updates
        self.action_size = env.action_size

        self.plan = jnp.zeros((self.n_steps, self.action_size))

        # lr_schedule = optax.cosine_decay_schedule(
        #     init_value=lr,
        #     decay_steps=n_updates,
        #     alpha=0.1
        # )
        #
        # self.optimizer = optax.chain(
        #     optax.clip_by_global_norm(5.0),
        #     optax.adam(learning_rate=lr_schedule)
        # )

        @jax.custom_vjp
        def safe_cost(plan, start_state):
            return self._trajectory_cost(plan, start_state)

        def f_fwd(plan, start_state):
            return safe_cost(plan, start_state), (plan, start_state)

        def f_bwd(res, g):
            # Backward pass
            plan, start_state = res
            grads = jax.jacfwd(self._trajectory_cost, argnums=0)(plan, start_state)
            jax.debug.print("cost {}", self._trajectory_cost(plan, start_state))
            return grads * g, None

        safe_cost.defvjp(f_fwd, f_bwd)

        # --- SOLVER SETUP ---
        self.solver = jaxopt.LBFGSB(
            fun=safe_cost,
            maxiter=n_updates,
            tol=1e-2,
            implicit_diff=False,
            value_and_grad=False,
            linesearch="backtracking",
            condition="armijo",
            maxls=10
        )

        # self.optimizer = optax.scale_by_lbfgs()

    def get_action(self, current_state, rng):
        # opt_state = self.optimizer.init(self.plan)

        lower_bounds = jnp.full_like(self.plan, -1.0)
        upper_bounds = jnp.full_like(self.plan, 1.0)

        results = self.solver.run(
            init_params=self.plan,
            bounds=(lower_bounds, upper_bounds),
            start_state=current_state
        )
        self.plan = results.params
        return self.plan, rng


        # def optimization_step(carry, step_index):
        #     current_plan, current_opt_state = carry
        #     grads = jax.jacfwd(self._trajectory_cost, argnums=0)(current_plan, current_state)
        #
        #     jax.debug.print("step {}, max grad {}, cost {}",
        #                     step_index,
        #                     jnp.max(jnp.abs(grads)),
        #                     self._trajectory_cost(current_plan, current_state))
        #
        #     updates, new_opt_state = self.optimizer.update(grads, current_opt_state, current_plan)
        #     new_plan = optax.apply_updates(current_plan, updates)
        #     new_plan = jnp.clip(new_plan, -1.0, 1.0)
        #
        #     return (new_plan, new_opt_state), None
        #
        # xs = jnp.arange(self.n_updates)
        #
        # (self.plan, _), _ = jax.lax.scan(
        #     optimization_step,
        #     (self.plan, opt_state),
        #     xs,
        #     length=self.n_updates
        # )

        return self.plan, rng


    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_cost(self, action_sequence, start_state):

        def step_fn(current_state, action):
            next_state = self.env.step(current_state, action)
            cost = -next_state.reward
            return next_state, cost

        final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)
        return jnp.sum(costs)