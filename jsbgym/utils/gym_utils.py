import gymnasium as gym
import numpy as np

class MyNormalizeObservation(gym.wrappers.NormalizeObservation):
    def __init__(self, env: gym.Env, eval: bool = False, epsilon: float = 1e-8):
        super().__init__(env, epsilon)
        self.eval = eval

    def get_obs_rms(self):
        return self.obs_rms

    def set_obs_rms(self, mean, var):
        self.obs_rms.mean = mean
        self.obs_rms.var = var

    def normalize(self, obs):
        if not self.eval:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)