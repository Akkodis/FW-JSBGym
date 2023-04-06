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

K_long: dict[str, float]
K_long, _, __ = uav.compute_long_pid_gains()
kp: float = K_long['kp_pitch']
kd: float = K_long['kd_pitch']

# airspeed closed loop transfer function (with PI)
num: list = [kp * uav.a_pitch3]
den: list = [1, uav.a_pitch1 + (kd * uav.a_pitch3), uav.a_pitch2 + (kp * uav.a_pitch3)]
H: ctl.TransferFunction = ctl.tf(num, den)
print(H)

t = np.linspace(0, 10, 101)
Va_ref_ = 1.0 * np.ones(t.shape)
_, y = ctl.forced_response(H, t, U=Va_ref_)

# plotting
plt.close('all')
plt.subplot(2 , 1 , 1)
plt.title('(control) Pitch Angle PD CLFR : no convergence to ref is normal according to the book')
plt.plot(t, y , 'blue')
plt.plot(t, Va_ref_, 'red')
plt.xlabel('t [s]')
plt.grid()
plt.legend(labels =('Va_', 'Va_ref_'))
# plt.subplot(2 , 1 , 2)
# plt.plot(t , u ,'green')
# plt.xlabel('t [ s ]')
# plt.grid()
# plt.legend(labels =('u',))

plt.show()
pass