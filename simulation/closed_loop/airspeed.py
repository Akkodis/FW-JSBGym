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
K_vth: dict[str, float] = uav.compute_long_pid_gains()
kp = K_vth['kp_vth']
# kp = 0.5
ki = K_vth['ki_vth']
# ki = 0.0

# airspeed closed loop transfer function (with PI)
num: list = [uav.av2 * kp, uav.av2 * ki]
den: list = [1, uav.av1 + uav.av2 * kp, uav.av2 * ki]
H: ctl.TransferFunction = ctl.tf(num, den)

print(H)

t = np.linspace(0, 10, 101)
Va_ref_ = 0.2 * np.ones(t.shape)
_, y = ctl.forced_response(H, t, U=Va_ref_)

# plotting
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
plt.figure(1 , figsize =(fig_width_cm /2.54 , fig_height_cm /2.54))
plt.subplot(2 , 1 , 1)
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