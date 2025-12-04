from .classes.normal import NormalEnv
from .options.rewards_list import *
from .options.done_list import *
from .classes.base_and_reward import RewardMJXEnv
import json

REWARD_MAP = {
    "speed": speed_reward,
    "ctrl": ctrl_reward,
    "speed_cost": speed_cost,
    "z_cost": z_cost,
    "quat_cost": quat_cost
}

TERMINAL_MAP = {
    "none": no_termination
}

XML_MAP = {
    "normal": "./envs/xmls/normal_plane.xml",
}

CLASS_MAP = {
    "normal": NormalEnv
}

def load_environment(robot_xml_string, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    env_name = config.get("env_name", "normal")
    xml_path = XML_MAP[env_name]
    load_type = config.get("load_type", "class")

    if load_type == "class":
        class_obj = CLASS_MAP[env_name]
        return class_obj(xml_path=xml_path,
                         robot_xml_string=robot_xml_string,
                         params=config,
                         n_frames=config["n_frames"])

    elif load_type == "combine":
        reward_names = config["reward_fns"]
        done_name = config.get("done_fn", "none")

        reward_fns = [REWARD_MAP[name] for name in reward_names]
        reward_weights = config.get("reward_weights", [1.0] * len(reward_fns))
        reward_weights = jnp.array(reward_weights)
        done_fn = TERMINAL_MAP[done_name]

        return RewardMJXEnv(xml_string=xml_path,
                            robot_xml_string=robot_xml_string,
                            params=config,
                            reward_fns=reward_fns,
                            reward_weights=reward_weights,
                            done_fn=done_fn,
                            n_frames=config["n_frames"])
    return None

