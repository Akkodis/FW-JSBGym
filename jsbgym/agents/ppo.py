import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from jsbgym.utils.gym_utils import MyNormalizeObservation


def make_env(env_id, config, render_mode, telemetry_file=None, eval=False, gamma=0.99, run_name='', idx=0):
    def thunk():
        env = gym.make(env_id, config_file=config, telemetry_file=telemetry_file,
                        render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = MyNormalizeObservation(env, eval=eval)
        if not eval:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # if the envs arg is a vector env, we need to get the single env action and obs space
        # usually used in parallized envs for training
        if isinstance(envs, gym.vector.SyncVectorEnv):
            self.single_action_space = envs.single_action_space
            self.single_obs_space = envs.single_observation_space
        else: # single env usually used for evaluation
            self.single_action_space = envs.action_space
            self.single_obs_space = envs.observation_space
        num_of_filters = 3 # default 3
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=num_of_filters, 
                                 kernel_size=(self.single_obs_space.shape[1], 1), stride=1)),
            nn.Tanh(),
            nn.Flatten()
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.single_obs_space.shape[2]*num_of_filters, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.single_obs_space.shape[2]*num_of_filters, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(self.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(self.conv(x))

    def get_action_std(self):
        return torch.exp(self.actor_logstd.expand_as(torch.zeros(1, np.prod(self.single_action_space.shape))))

    def get_action_and_value(self, x, action=None):
        conv_out = self.conv(x)
        action_mean = self.actor_mean(conv_out)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, action_mean, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(conv_out)
