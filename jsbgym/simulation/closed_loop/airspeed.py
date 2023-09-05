import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import control as ctl
import numpy as np
import matplotlib.pyplot as plt
from jsbgym.models.aerodynamics import AeroModel
from jsbgym.trim.trim_point import TrimPoint

trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)
K_long: dict[str, float]
K_long, _, __ = uav.compute_long_pid_gains()
kp = K_long['kp_vth']
# kp = 0.5
ki = K_long['ki_vth']
# ki = 0.0

# airspeed closed loop transfer function (with PI)
num: list = [uav.av2 * kp, uav.av2 * ki]
den: list = [1, uav.av1 + uav.av2 * kp, uav.av2 * ki]
H: ctl.TransferFunction = ctl.tf(num, den)

print(H)

t = np.linspace(0, 10, 101)
Va_ref_ = 6.3 * np.ones(t.shape)
_, y = ctl.forced_response(H, t, U=Va_ref_)

# plotting
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
plt.subplot(2 , 1 , 1)
plt.title('Airspeed closed loop (PI) forced response (via python control pkg)')
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