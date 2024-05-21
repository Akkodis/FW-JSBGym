import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# ALGO LOGIC: initialize agent here:
class SoftQNetwork_SAC(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor_SAC(nn.Module):
    def __init__(self, env):
        super().__init__()
        # if the envs arg is a vector env, we need to get the single env action and obs space
        # usually used in parallized envs for training
        if isinstance(env, gym.vector.SyncVectorEnv):
            self.single_action_space = env.single_action_space
            self.single_obs_space = env.single_observation_space
            self.action_space_high = env.action_space.high
            self.action_space_low = env.action_space.low
        else: # single env usually used for evaluation
            self.single_action_space = env.action_space
            self.single_obs_space = env.observation_space
            self.action_space_high = np.expand_dims(env.action_space.high, axis=0)
            self.action_space_low = np.expand_dims(env.action_space.low, axis=0)

        self.fc1 = nn.Linear(np.array(self.single_obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(self.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(self.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((self.action_space_high - self.action_space_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((self.action_space_high + self.action_space_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean