import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import control
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint
from agents.pid import PID


trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)


nums: list = [[ [uav.av2], [-uav.av3], [1] ]]
den: list = [1, uav.av1]
dens: list = [ [den, den, den] ]
H: control.TransferFunction = control.tf(nums, dens)
print(H)

t = np.linspace(0, 10, 101)

# step response
# _, y0 = control.step_response(H, t, input=0, squeeze=True)

# forced response
# dt_: np.ndarray = 0.2 * np.ones(t.shape)
# theta_: np.ndarray = np.zeros(t.shape)
# dv: np.ndarray = np.zeros(t.shape)
# u: np.ndarray = np.array([dt_, theta_, dv]) # commands : steps
# _, y = control.forced_response(H, t, U=u, squeeze=True)

# print last output
# print(y[-1])

# # plotting
# plt.close('all')
# fig_width_cm = 24
# fig_height_cm = 18
# plt.figure(1 , figsize =(fig_width_cm /2.54 , fig_height_cm /2.54))
# plt.subplot(2 , 1 , 1)
# plt.plot(t , y ,'blue')
# # plt.xlabel.'t [ s ]')
# plt.grid()
# plt.legend(labels =('y',))
# # plt.subplot(2 , 1 , 2)
# # plt.plot(t , u ,'green')
# # plt.xlabel('t [ s ]')
# # plt.grid()
# # plt.legend(labels =('u',))

# plt.show()

Va_ref: float = 0.2 # set reference as deviation from Va_trim m/s
Va_: float = 0.0 # initial value for Va_

# compute pid gains airspeed hold with commanded throttle v_th
K_vth: dict[str, float] = uav.compute_long_pid_gains()
vth_pid: PID = PID(kp=K_vth['kp_vth'], ki=K_vth['ki_vth'], kd=0, dt=1/120, limit=uav.throttle_limit, is_throttle=True)
vth_pid.set_reference(Va_ref)
cmd_th: float = vth_pid.update(Va_, normalize=True)

dt_: np.ndarray = cmd_th * np.ones(t.shape)
theta_: np.ndarray = np.zeros(t.shape)
dv: np.ndarray = np.zeros(t.shape)
u: np.ndarray = np.array([dt_, theta_, dv]) # commands : steps
_, y = control.forced_response(H, t, U=u, squeeze=True)

print(y[-1])

# plotting
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
plt.figure(1 , figsize =(fig_width_cm /2.54 , fig_height_cm /2.54))
plt.subplot(2 , 1 , 1)
plt.plot(t , y ,'blue')
plt.grid()
plt.legend(labels =('y',))

plt.show()