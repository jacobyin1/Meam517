import optax
import jax
import jaxopt
import jax.numpy as jnp
from functools import partial


class Shooting:
    def __init__(self, env, n_steps: int = 10, n_updates: int = 20, config=None):
        if config is not None:
            n_steps = config.get("n_steps", n_steps)
            n_updates = config.get("n_updates", n_updates)

        self.env = env
        self.n_steps = n_steps
        self.n_updates = n_updates
        self.action_size = env.action_size

        self.plan = jnp.zeros((self.n_steps, self.action_size))
        self.best_plan = self.plan.copy()
        self.best_cost = jnp.array(jnp.inf)

        self.opt_type = config.get("optimizer", "lbfgs") if config is not None else "lbfgs"

        if self.opt_type == "gradient":
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=config["lr"] if config is not None else 1e-3)
            )

        # @jax.custom_vjp
        # def safe_cost(plan, start_state):
        #     return self._trajectory_cost(plan, start_state)
        #
        # def f_fwd(plan, start_state):
        #     return safe_cost(plan, start_state), (plan, start_state)
        #
        # def f_bwd(res, g):
        #     # Backward pass
        #     plan, start_state = res
        #     grads = jax.jacfwd(self._trajectory_cost, argnums=0)(plan, start_state)
        #     jax.debug.print("here")
        #     return grads * g, None
        #
        # safe_cost.defvjp(f_fwd, f_bwd)

        if self.opt_type == "lbfgs":
            self.solver = jaxopt.LBFGSB(
                fun=self._trajectory_cost,
                maxiter=n_updates,
                tol=1e-2,
                implicit_diff=False,
                value_and_grad=False,
                linesearch="backtracking",
                condition="armijo",
                maxls=10
            )


    def get_action(self, current_state, rng):
        lower_bounds = jnp.full_like(self.plan, -1.0)
        upper_bounds = jnp.full_like(self.plan, 1.0)

        self.plan = jax.random.normal(key=rng, shape=self.plan.shape) * 0.5

        if self.opt_type == "lbfgs":
            results = self.solver.run(
                init_params=self.plan,
                bounds=(lower_bounds, upper_bounds),
                start_state=current_state
            )

            self.plan = results.params

            info = {
                "cost": results.state.value,
                "error": results.state.error,
                "iters": results.state.iter_num
            }

        if self.opt_type == "gradient":
            def optimization_step(carry, step_index):
                current_plan, current_opt_state, best_plan, best_cost = carry
                cost, grads = jax.value_and_grad(self._trajectory_cost, argnums=0)(current_plan, current_state)

                is_better = cost < best_cost
                new_best_cost = jnp.where(is_better, cost, best_cost)
                new_best_plan = jnp.where(is_better, current_plan, best_plan)

                jax.debug.print("step {}, max grad {}, cost {}",
                                step_index,
                                jnp.max(jnp.abs(grads)),
                                cost)

                updates, new_opt_state = self.optimizer.update(grads, current_opt_state, current_plan)
                new_plan = optax.apply_updates(current_plan, updates)
                new_plan = jnp.clip(new_plan, lower_bounds, upper_bounds)

                info = {
                    "step": step_index,
                    "cost": cost,
                    "max grad": jnp.max(jnp.abs(grads)),
                }

                return (new_plan, new_opt_state, new_best_plan, new_best_cost), info

            xs = jnp.arange(self.n_updates)
            init_opt_state = self.optimizer.init(self.plan)
            (self.plan, _, self.best_plan, self.best_cost), info = jax.lax.scan(
                optimization_step,
                (self.plan, init_opt_state, self.best_plan, self.best_cost),
                xs,
                length=self.n_updates
            )

        return self.best_plan, info, rng


    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_cost(self, action_sequence, start_state):
        self.env.reset(jax.random.PRNGKey(0))
        def step_fn(current_state, action):
            next_state = self.env.step(current_state, action)
            cost = -next_state.reward
            return next_state, cost

        final_state, costs = jax.lax.scan(step_fn, start_state, action_sequence)

        diffs = action_sequence[1:] - action_sequence[:-1]
        smooth_cost = jnp.sum(jnp.square(diffs))

        return jnp.sum(costs) + 0.1 * smooth_cost