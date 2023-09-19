import gymnasium as gym
import torch
from jsbgym.trim.trim_point import TrimPoint
import numpy as np
import agents.ppo as ppo

env = gym.make('JSBSim-AttitudeControlTask-v0', config='config/bohn_ppo.yaml', render_mode='plot')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='fgear_plot_scale')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='none')
obs = env.reset(seed=1)
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

ppo_agent = ppo.Agent(envs=env)
ppo_agent.load_state_dict(torch.load('models/AttitudeControlTaskEnv-v0__ppo_continuous_action__1__1695119387.pt'))
ppo_agent.eval()

for _ in range(1200):
    # action = env.action_space.sample()
    action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    action = ppo_agent.get_action_and_value(torch.from_numpy(obs))[0].detach().numpy()
    obs, reward, truncated, terminated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset(seed=1)

env.close()