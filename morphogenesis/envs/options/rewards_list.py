import jax.numpy as jnp
import jax

def speed_reward(state, action):
    vel_reward = 5.0 * state.qvel[0] + 5.0 * state.qvel[1]
    # jax.debug.print("Velocity Reward: {vel_reward}", vel_reward=3 * vel_reward)
    return 5 * vel_reward

def ctrl_cost(state, action):
    ctrl_cost = -0.1 * jnp.sum(action ** 2)
    return ctrl_cost

def speed_cost(state, action):
    vel_cost = -.01 * (jnp.linalg.norm(state.qvel[6:] ) - 3) ** 2
    # jax.debug.print("Velocity Cost: {vel_cost}", vel_cost=vel_cost)
    return vel_cost

def z_cost(state, action):
    z_cost = (-3.0 * (state.qpos[2] - 0.5) ** 2)
    # jax.debug.print("Z Cost: {z_cost}", z_cost=z_cost)
    return z_cost

def quat_cost(state, action):
    quat = state.qpos[3:7]
    uprightness = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    quat_cost = -1 * (1 - uprightness)
    # jax.debug.print("Quat Cost: {quat_cost}", quat_cost=quat_cost)
    return quat_cost