import jax.numpy as jnp

def speed_reward(state, action):
    vel_reward = 10.0 * state.qvel[0] + 10.0 * state.qvel[1]
    return vel_reward

def ctrl_reward(state, action):
    ctrl_cost = -0.1 * jnp.sum(action ** 2)
    return ctrl_cost

def speed_cost(state, action):
    vel_cost = -1 * jnp.linalg.norm(state.qvel[6:])
    return vel_cost

def z_cost(state, action):
    z_cost = -0.5 * state.qpos[2] ** 2
    return z_cost

def quat_cost(state, action):
    quat = state.qpos[3:7]
    uprightness = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    quat_cost = -0.2 * (1 - uprightness)
    return quat_cost