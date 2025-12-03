import jax.numpy as jnp

def speed_reward(state, action):
    vel_reward = 5.0 * state.qvel[0]
    return vel_reward

def ctrl_reward(state, action):
    ctrl_cost = -0.1 * jnp.sum(action ** 2)
    return ctrl_cost

