import json
import jax
import numpy as np


class JaxEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (jax.Array, np.ndarray)):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def save_log(filename, actions, states, info):

    actions = jax.device_get(actions)
    states = jax.device_get(states)
    info = jax.device_get(info)

    state_dict = {
        "qpos": states.pipeline_state.qpos,
        "qvel": states.pipeline_state.qvel,
        "reward": states.reward
    }

    output_data = {
        "config": {
            "file": filename
        },
        "info": info,
        "actions": actions,
        "trajectory": state_dict
    }

    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        json.dump(output_data, f, cls=JaxEncoder, indent=4)
    print("Done.")