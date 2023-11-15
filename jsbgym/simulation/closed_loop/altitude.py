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
k_dc_pitch: float
K_long, _, k_dc_pitch = uav.compute_long_pid_gains()
kp_h: float = K_long['kp_h']
ki_h: float = K_long['ki_h']

# airspeed closed loop transfer function (with PI)
num: list = [k_dc_pitch * uav.Va_trim * kp_h, k_dc_pitch * uav.Va_trim * kp_h * (ki_h / kp_h)]
den: list = [1, k_dc_pitch * uav.Va_trim * kp_h, k_dc_pitch * uav.Va_trim * ki_h]
H: ctl.TransferFunction = ctl.tf(num, den)
print(H)

t = np.linspace(0, 10, 101)
h_ref = 1.0 * np.ones(t.shape)
_, y = ctl.forced_response(H, t, U=h_ref)

# plotting
plt.close('all')
plt.title('(control) Altitude h hold (outer loop)')
plt.plot(t, y , 'blue')
plt.plot(t, h_ref, 'red')
plt.xlabel('t [s]')
plt.grid()
plt.legend(labels =('altitude', 'altitude_ref'))

plt.show()
pass