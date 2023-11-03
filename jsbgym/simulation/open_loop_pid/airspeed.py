import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import numpy as np
import matplotlib.pyplot as plt
from jsbgym.models.aerodynamics import AeroModel
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.agents.pid import PID


def integrate_ss(A, B, X, U, dt):
    X_out: float = np.exp(A * dt) * X + dt * np.exp(A * 0) * B * U
    return X_out

trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# matrices A et B
A: float = -uav.av1
B: float = uav.av2

# timestep (120Hz)
dt: float = 1/120

# initial value of Va_
Va = 0.0

# reference value of Va
Va_ref = 6.0

# arrays for plotting
Va_array: list = [Va]
cmds_th: list = []
errs: list = []

# compute pid gains airspeed hold with commanded throttle v_th
K_long: dict[str, float]
K_long, _, __ = uav.compute_long_pid_gains()

# comment / uncomment as desired : use computed pid_gains or use hardcoded, found by hand values
kp_vth: float = K_long['kp_vth']
ki_vth: float = K_long['ki_vth']
# kp_vth: float = 1.0
# ki_vth: float = 0.1
vth_pid: PID = PID(kp=kp_vth, ki=ki_vth, kd=0, dt=dt, trim=trim, limit=uav.throttle_limit, is_throttle=True)

# faire tourne la boucle pid pendant 10 secondes à 120Hz
tsteps = 10*120
t_pid: np.ndarray = np.linspace(0, 10, tsteps+1) # +1 car on a Va à 0.0 au début (plotting)
for i in range (0, tsteps):
    cmd_th_: float
    err: float
    vth_pid.set_reference(Va_ref) # set reference
    cmd_th, err = vth_pid.update(Va, saturate=True) # get command and error
    cmds_th.append(cmd_th) # append command : plots
    errs.append(err) # append error : plots

    # integrate the model and get the next Va state
    Va = integrate_ss(A, B, Va, cmd_th, dt)
    Va_array.append(Va) # append Va : plots

plt.subplot(3, 1, 1)
plt.title('Coded PID with saturation [0,1] with FTBO sim (custom implementation of integration)')
plt.plot(t_pid, Va_array, 'blue')
plt.plot(t_pid, Va_ref * np.ones(t_pid.shape), 'red')
plt.legend(labels=('Va', 'Va_ref'))
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_pid[0:tsteps], cmds_th, 'green')
plt.legend(labels=('cmd_th',))
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_pid[0:tsteps], errs, 'black')
plt.legend(labels=('err',))
plt.xlabel('t [s]')
plt.grid()

plt.show()
