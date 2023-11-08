import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5,1), stride=1)), # input ?x5x12x1, output ?x1x12x3
            nn.Tanh(),
            nn.Flatten()
        )
        self.critic = nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(12*3, 64)), # 12 is the number of features extracted by 1 conv * num of conv filters
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(12*3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(self.conv(x))

    def get_action_mean(self, x):
        return self.actor_mean(self.conv(x))

    def get_action_and_value(self, x, action=None, eval=False):
        conv_out = self.conv(x)
        action_mean = self.actor_mean(conv_out)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        print(action_std)
        probs = Normal(action_mean, action_std)
        if action is None:
            if eval: # when evaluating, we ignore the stochasticity of the policy
                action = action_mean
            else:
                action = probs.sample()
        return action, action_mean, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(conv_out)

