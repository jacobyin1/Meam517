import jax.numpy as jnp
import jax
import jaxopt
import mujoco
import mujoco.mjx as mjx


def get_endpoints_from_geom(pos, size, quat):
    radius = size[0]
    half_length = size[1]
    local_z = jnp.array([0., 0., 1.])

    q_w = quat[0]
    q_xyz = quat[1:]

    t = 2 * jnp.cross(q_xyz, local_z)
    direction = local_z + q_w * t + jnp.cross(q_xyz, t)
    offset = direction * half_length

    start_pos = pos - offset
    end_pos = pos + offset

    return start_pos, end_pos, radius

def calculate_fromto_params(start_pos, end_pos, radius):
    pos = (start_pos + end_pos) / 2.0
    diff = end_pos - start_pos
    length = jnp.linalg.norm(diff)
    size = jnp.array([radius, length / 2.0, 0.0])

    vec = diff / (length + 1e-6)  # Normalize target vector
    ref = jnp.array([0., 0., 1.])  # Reference vector (Capsule default)

    v = jnp.cross(ref, vec)
    xyz = v
    w = 1.0 + jnp.dot(ref, vec)

    quat = jnp.concatenate([jnp.array([w]), xyz])
    quat = quat / (jnp.linalg.norm(quat) + 1e-6)

    return pos, size, quat


class ShootingParams:

    def __init__(self, env, n_steps: int = 30, n_updates: int = 10, config=None):
        if config is not None:
            n_steps = config.get("n_steps", n_steps)
            n_updates = config.get("n_updates", n_updates)

        self.env = env
        self.n_steps = n_steps
        self.n_updates = n_updates
        self.action_size = env.action_size


        self.plan = jnp.zeros((self.n_steps, self.action_size))

        model = env._mj_model # noqa

        excluded_world_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
        excluded_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.included_ids = list(range(model.nbody))
        self.included_ids.remove(excluded_world_id)
        self.included_ids.remove(excluded_torso_id)

        new_included_ids = self.included_ids.copy()
        self.parent_ids = []
        self.duplicates = {}
        for child_id in self.included_ids:
            parent_id = model.body_parentid[child_id]
            if parent_id == excluded_world_id or parent_id == excluded_torso_id:
                new_included_ids.remove(child_id)
            elif parent_id in self.parent_ids:
                new_included_ids.remove(child_id)
                self.duplicates[child_id] = self.included_ids[self.parent_ids.index(parent_id)]
            else:
                self.parent_ids.append(parent_id)
        self.included_ids = new_included_ids

        self.end_ids = []
        for child_id in self.included_ids:
            if child_id not in self.parent_ids:
                self.end_ids.append(child_id)

        self.body_to_parent_geom = {}
        for i, bid in enumerate(self.parent_ids):
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    self.body_to_parent_geom[self.included_ids[i]] = gid

        self.end_to_geom = {}
        for eid in self.end_ids:
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == eid:
                    self.end_to_geom[eid] = gid

        self.body_shifts = jnp.zeros((len(self.included_ids) + len(self.end_ids), 3))
        self._setup_solver()

    def update_sys(self, sys, shifts):
        new_pos = sys.body_pos
        new_geom_size = sys.geom_size
        new_geom_pos = sys.geom_pos
        new_geom_quat = sys.geom_quat

        for i, body_id in enumerate(self.included_ids):
            old_pos = new_pos.at[body_id].get()
            new_pos = new_pos.at[body_id].set(shifts[i] + old_pos)

            old_from, old_to, radius = get_endpoints_from_geom(
                sys.geom_pos[self.body_to_parent_geom[body_id]],
                sys.geom_size[self.body_to_parent_geom[body_id]],
                sys.geom_quat[self.body_to_parent_geom[body_id]]
            )

            new_from = old_from
            new_to = old_to + shifts[i]

            pos, size, quat = calculate_fromto_params(new_from, new_to, radius)
            new_geom_pos = new_geom_pos.at[self.body_to_parent_geom[body_id]].set(pos)
            new_geom_size = new_geom_size.at[self.body_to_parent_geom[body_id]].set(size)
            new_geom_quat = new_geom_quat.at[self.body_to_parent_geom[body_id]].set(quat)

            for dup_id, real_id in self.duplicates.items():
                if body_id == real_id:
                    old_pos = new_pos.at[dup_id].get()
                    new_pos = new_pos.at[dup_id].set(shifts[i] + old_pos)

        for i, body_id in enumerate(self.end_ids):
            old_from, old_to, radius = get_endpoints_from_geom(
                sys.geom_pos[self.end_to_geom[body_id]],
                sys.geom_size[self.end_to_geom[body_id]],
                sys.geom_quat[self.end_to_geom[body_id]]
            )

            new_from = old_from
            new_to = old_to + shifts[len(self.included_ids) + i]

            pos, size, quat = calculate_fromto_params(new_from, new_to, radius)
            new_geom_pos = new_geom_pos.at[self.end_to_geom[body_id]].set(pos)
            new_geom_size = new_geom_size.at[self.end_to_geom[body_id]].set(size)
            new_geom_quat = new_geom_quat.at[self.end_to_geom[body_id]].set(quat)

        return sys.tree_replace({
            'body_pos': new_pos,
            'geom_pos': new_geom_pos,
            'geom_size': new_geom_size,
            'geom_quat': new_geom_quat
        })

    def joint_cost(self, params):
        """
        Evaluates the cost of a (Design, Control) pair.
        """
        shifts = params['shifts']
        controls = params['controls']

        optimized_sys = self.update_sys(self.env.sys, shifts)
        rng = jax.random.PRNGKey(0)
        data = mjx.make_data(optimized_sys)

        def step_fn(carry, action):
            d, _ = carry

            def physics_loop(d_inner, _):
                d_inner = d_inner.replace(ctrl=action)
                return mjx.step(optimized_sys, d_inner), None

            d_next, _ = jax.lax.scan(physics_loop, d, None, length=self.env.n_frames)

            reward = self.env.compute_reward(d_next, action)
            step_cost = -reward
            return (d_next, rng), step_cost

        _, step_costs = jax.lax.scan(step_fn, (data, rng), controls)
        design_reg = 0.1 * jnp.sum(shifts ** 2)

        return jnp.sum(step_costs) + design_reg

    def _setup_solver(self):
        @jax.custom_vjp
        def safe_cost(params):
            return self.joint_cost(params)

        def f_fwd(params):
            return safe_cost(params), params

        def f_bwd(res, g):
            # Backward pass
            grads = jax.jacfwd(self.joint_cost, argnums=0)(res)
            grads_scaled = jax.tree_util.tree_map(lambda x: x * g, grads)
            jax.debug.print("cost {}", self.joint_cost(res))
            return (grads_scaled,)

        safe_cost.defvjp(f_fwd, f_bwd)

        self.solver = jaxopt.LBFGSB(
            fun=safe_cost,
            maxiter=self.n_updates,
            tol=1e-2,
            implicit_diff=False,
            value_and_grad=False,
            linesearch="backtracking",
            condition="armijo",
            maxls=10
        )

    def optimize(self):
        init_params = {
            "shifts": self.body_shifts,
            "controls": self.plan
        }

        shift_bounds_low = jnp.full_like(self.body_shifts, -0.2)
        shift_bounds_high = jnp.full_like(self.body_shifts, 0.2)
        ctrl_bounds_low = jnp.full_like(self.plan, -1.0)
        ctrl_bounds_high = jnp.full_like(self.plan, 1.0)

        lower_bounds = {"shifts": shift_bounds_low, "controls": ctrl_bounds_low}
        upper_bounds = {"shifts": shift_bounds_high, "controls": ctrl_bounds_high}

        sol = self.solver.run(init_params, bounds=(lower_bounds, upper_bounds))

        self.body_shifts = sol.params['shifts']
        self.plan = sol.params['controls']

        return self.body_shifts, self.plan