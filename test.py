from envs.jsbsim_env import JSBSimEnv
from envs.tasks import AttitudeControlTask
from trim.trim_point import TrimPoint
import numpy as np

env = JSBSimEnv(task_type=AttitudeControlTask, render_mode='plot_scale')
obs = env.reset()
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

for _ in range(1000):
    # action = env.action_space.sample()
    action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    obs, reward, done, info = env.step(action)
    if done:
        break

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np
# import threading
# import random
# import time

# class MyDataClass():

#     def __init__(self):

#         self.XData = [0]
#         self.YData = [0]


# class MyPlotClass():

#     def __init__(self, dataClass):

#         self._dataClass = dataClass

#         self.hLine, = plt.plot(0, 0)

#         self.ani = FuncAnimation(plt.gcf(), self.run, interval = 1000, repeat=True)


#     def run(self, i):  
#         print("plotting data")
#         self.hLine.set_data(self._dataClass.XData, self._dataClass.YData)
#         self.hLine.axes.relim()
#         self.hLine.axes.autoscale_view()


# class MyDataFetchClass(threading.Thread):

#     def __init__(self, dataClass):

#         threading.Thread.__init__(self)

#         self._dataClass = dataClass
#         self._period = 0.25
#         self._nextCall = time.time()


#     def run(self):

#         while True:
#             print("updating data")
#             # add data to data class
#             self._dataClass.XData.append(self._dataClass.XData[-1] + 1)
#             self._dataClass.YData.append(random.randint(0, 256))
#             # sleep until next execution
#             self._nextCall = self._nextCall + self._period;
#             time.sleep(self._nextCall - time.time())


# data = MyDataClass()
# plotter = MyPlotClass(data)
# fetcher = MyDataFetchClass(data)

# fetcher.start()
# plt.show()