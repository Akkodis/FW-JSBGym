import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import control
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint


trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# av1, av2, av3 TF coeffs
av1: float = ((uav.rho * trim.Va_ms * uav.S) / uav.mass) / (uav.CDo + uav.CDalpha * trim.alpha_rad + uav.CDde * trim.elevator)
av2: float = (uav.Pwatt / uav.Khp2w) * uav.Khp2ftlbsec
av3: float = uav.G * np.cos(trim.theta_rad - trim.alpha_rad)

nums: list = [[ [av2], [-av3], [1] ]]
den: list = [1, av1]
dens: list = [ [den, den, den] ]
H: control.TransferFunction = control.tf(nums, dens)
print(H)

t = np.linspace(0, 10, 101)

# step response
# _, y0 = control.step_response(H, t, input=0, squeeze=True)

# forced response
dt_: np.ndarray = 0.2 * np.ones(t.shape)
theta_: np.ndarray = np.zeros(t.shape) # np.ones(t.shape)
dv: np.ndarray = np.zeros(t.shape) # np.ones(t.shape)
u: np.ndarray = np.array([dt_, theta_, dv]) # commands : steps
_, y = control.forced_response(H, t, U=u, squeeze=True)

# print last output
print(y[-1])

# plotting
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
plt.figure(1 , figsize =(fig_width_cm /2.54 , fig_height_cm /2.54))
plt.subplot(2 , 1 , 1)
plt.plot(t , y ,'blue')
# plt.xlabel.'t [ s ]')
plt.grid()
plt.legend(labels =('y',))
# plt.subplot(2 , 1 , 2)
# plt.plot(t , u ,'green')
# plt.xlabel('t [ s ]')
# plt.grid()
# plt.legend(labels =('u',))

plt.show()

Va_ref: float = 0.2 # set reference as deviation from Va_trim m/s
Va_: float = 0.0 # initial value for Va_