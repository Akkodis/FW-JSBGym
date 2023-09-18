from jsbgym.envs.jsbsim_env import JSBSimEnv
from jsbgym.envs.attitude_control import AttitudeControlTaskEnv
from jsbgym.trim.trim_point import TrimPoint
import numpy as np

env = AttitudeControlTaskEnv(config_file="config/bohn_ppo.yaml", render_mode='plot_scale')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='fgear_plot_scale')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='none')
obs = env.reset()
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

terminated = False
truncated = False
while not (terminated or truncated):
    # action = env.action_space.sample()
    action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    obs, reward, truncated, terminated, info = env.step(action)
    # print(info)
