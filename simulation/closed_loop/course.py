import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import control as ctl
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint


trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

K_lat: dict[str, float]
K_lat, _ = uav.compute_lat_pid_gains()
kp: float = K_lat['kp_course']
ki: float = K_lat['ki_course']

# roll closed loop transfer function (with PID)
num: list = [kp * (uav.G/uav.Va_trim), ki * (uav.G/uav.Va_trim)]
den: list = [1, kp * (uav.G/uav.Va_trim), ki * (uav.G/uav.Va_trim)]
H: ctl.TransferFunction = ctl.tf(num, den)
print(H)

t = np.linspace(0, 10, 10*60)
course_ref = 45.0 * (np.pi/180) * np.ones(t.shape)
_, y = ctl.forced_response(H, t, U=course_ref)

# print roll final value
print(f'roll final value: {y[-1]} rad')

# plotting
plt.close('all')
plt.title('(control-FTBF) Course Angle PID outer loop')
plt.plot(t, y , 'blue')
plt.plot(t, course_ref, 'red')
plt.ylabel('[rad]')
plt.grid()
plt.legend(labels =('course', 'course_ref'))

plt.show()
