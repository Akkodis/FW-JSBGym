from jsbgym.envs.jsbsim_env import JSBSimEnv
from jsbgym.envs.tasks import AttitudeControlTask
from jsbgym.trim.trim_point import TrimPoint
import numpy as np

env = JSBSimEnv(episode_time_s=15, render_mode='plot_scale')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='fgear_plot_scale')
# env = JSBSimEnv(task_type=AttitudeControlTask, episode_time_s=15, render_mode='none')
obs = env.reset()
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

done = False
while not done:
    action = env.action_space.sample()
    # action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    obs, reward, done, info = env.step(action)
    # print(info)
