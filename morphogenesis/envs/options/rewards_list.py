import jax.numpy as jnp
import jax

def speed_reward(state, action):
    vel_reward = jax.nn.sigmoid(5.0 * state.qvel[0] + 5.0 * state.qvel[1])
    return 3 * vel_reward

def ctrl_cost(state, action):
    ctrl_cost = -0.1 * jnp.sum(action ** 2)
    return ctrl_cost

def speed_cost(state, action):
    vel_cost = -.01 * (jnp.linalg.norm(state.qvel[6:] ) - 1) ** 2
    return vel_cost

def z_cost(state, action):
    z_cost = -1.0 * state.qpos[2] ** 2
    return z_cost

def quat_cost(state, action):
    quat = state.qpos[3:7]
    uprightness = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    quat_cost = -0.3 * (1 - uprightness)
    return quat_cost