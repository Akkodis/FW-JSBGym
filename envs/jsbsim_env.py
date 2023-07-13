import gymnasium as gym
from gymnasium import spaces

class JSBSimEnv(gym.Env):
    metadata = {"render_modes": ["plot", "flightgear"]}
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box()