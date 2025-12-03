import jax
from brax.envs import env


class BaseMJXCrawler(env.Env):
    """
    Standard Brax/MJX boilerplate.
    Handles resetting and basic physics stepping.
    """

    def __init__(self, xml_path, params):
        super().__init__()
        self.xml_path = xml_path
        self.params = params
        # Load the specific XML for this environment
        self.sys = mjx.load(xml_path)

    def reset(self, rng):
        # Standard reset logic...
        pass

    def step(self, state, action):
        # Standard physics step
        next_pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # CALCULATE SPECIFIC REWARD (Polymorphism!)
        reward = self.compute_reward(next_pipeline_state, action)

        return state.replace(pipeline_state=next_pipeline_state), reward

    def compute_reward(self, state, action):
        raise NotImplementedError("Subclasses must implement this!")