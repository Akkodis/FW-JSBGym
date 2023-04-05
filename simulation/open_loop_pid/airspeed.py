import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint
from agents.pid import PID


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
Va_ = 0.0

# reference value of Va_
Va_ref = 6.0

# arrays for plotting
Va_array: list = [Va_]
cmds_th_: list = []
errs: list = []

# compute pid gains airspeed hold with commanded throttle v_th
K_vth: dict[str, float] = uav.compute_long_pid_gains()

# comment / uncomment as desired : use computed pid_gains or use hardcoded, found by hand values
# vth_pid: PID = PID(kp=K_vth['kp_vth'], ki=K_vth['ki_vth'], kd=0, dt=dt, limit=uav.throttle_limit, is_throttle=True)
vth_pid: PID = PID(kp=1.0, ki=0.1, kd=0, dt=dt, limit=uav.throttle_limit, is_throttle=True)

# faire tourne la boucle pid pendant 10 secondes à 120Hz
tsteps = 10*120
t_pid: np.ndarray = np.linspace(0, 10, tsteps+1) # +1 car on a Va_ à 0.0 au début (plotting)
for i in range (0, tsteps):
    cmd_th_: float
    err: float
    vth_pid.set_reference(Va_ref) # set reference
    cmd_th_, err = vth_pid.update(Va_, saturate=True) # get command and error
    cmds_th_.append(cmd_th_) # append command : plots
    errs.append(err) # append error : plots

    # integrate the model and get the next Va_ state
    Va_ = integrate_ss(A, B, Va_, cmd_th_, dt)
    Va_array.append(Va_) # append Va_ : plots

plt.subplot(3, 1, 1)
plt.title('Coded PID with saturation [0,1] with FTBO sim (custom implementation of integration)')
plt.plot(t_pid, Va_array, 'blue')
plt.plot(t_pid, Va_ref * np.ones(t_pid.shape), 'red')
plt.legend(labels=('Va_', 'Va_ref'))
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_pid[0:tsteps], cmds_th_, 'green')
plt.legend(labels=('cmd_th_',))
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_pid[0:tsteps], errs, 'black')
plt.legend(labels=('err',))
plt.xlabel('t [s]')
plt.grid()

plt.show()
