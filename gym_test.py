from envs.jsbsim_env import JSBSimEnv
from envs.tasks import AttitudeControlTask
from trim.trim_point import TrimPoint
import numpy as np

env = JSBSimEnv(task_type=AttitudeControlTask, render_mode='fgear_plot_scale')
obs = env.reset()
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

for _ in range(1000):
    # action = env.action_space.sample()
    action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    obs, reward, done, info = env.step(action)
    if done:
        break
