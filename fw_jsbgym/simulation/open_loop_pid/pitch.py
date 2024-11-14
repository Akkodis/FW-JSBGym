import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import control as ctl
import numpy as np
import matplotlib.pyplot as plt
from fw_jsbgym.models.aerodynamics import AeroModel
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.agents.pid import PID
from scipy import linalg

def integrate_ss(A: np.ndarray, B: np.ndarray, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    X_out: np.ndarray = linalg.expm(A * dt) @ X + dt * linalg.expm(A * 0) @ B @ U
    return X_out

trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# matrices A et B
# mat A : dynamique du pitch
A: np.ndarray = np.array([[0            , 1            ],
                          [-uav.a_pitch2, -uav.a_pitch1]])

# mat B : commande du pitch
B: np.ndarray = np.array([[0, 0           ],
                          [0, uav.a_pitch3]])

# O and cmd_de = 0 (elevator deflection) for initial command
cmd_de: float = 0.0
U: np.ndarray = np.array([[0],
                         [cmd_de]])

# vecteur d'état : pitch et pitch_dot (0,0) pour la valeur initiale
X: np.ndarray = np.array([[0],
                          [0]])

# vecteur dérivée de l'état : pitch_dot et pitch_dot_dot (0,0) pour la valeur initiale
X_dot: np.ndarray = np.array([[0],
                              [0]])

# timestep (120Hz)
dt: float = 1/120

# reference : pitch_ref = une ref et pitch_dot_ref = 0 -> on veut vitesse nulle à l'arrivée
pitch_ref: float = 45.0 * (np.pi / 180) # deg to rad
# pitch_ref: float = 1.0 # for step response testing
X_ref: np.ndarray = np.array([[pitch_ref],
                              [0        ]])

# arrays for plotting, initializing them with the initial value of X, U and X_ref - X
X_arr: np.array = np.array(X)
U_arr: np.array = np.array(U)
err_arr: np.array = np.array([X_ref[0,0] - X[0,0]])

# PID gains
K_long: dict[str, float]
K_long, _, __ = uav.compute_long_pid_gains()
kp_pitch: float = K_long['kp_pitch']
kd_pitch: float = K_long['kd_pitch']
pitch_pid: PID = PID(kp=kp_pitch, ki=0, kd=kd_pitch, dt=dt, limit=uav.elevator_limit, is_throttle=False)

tsteps: int = 10*120
t_pid: np.ndarray = np.linspace(0, 10, tsteps+1) # +1 car on a une val initiale dans X_arr au début (plotting)
for i in range (0, tsteps):
    err: float
    pitch_pid.set_reference(X_ref[0,0]) # set reference
    U[1], err = pitch_pid.update(state=X[0,0], state_dot=X[1,0], saturate=True) # get command and error
    U_arr = np.append(U_arr, U, axis=1) # append U : plots
    err_arr = np.append(err_arr, err) # append error : plots

    # integrate the model and get the next X state
    X = integrate_ss(A, B, X, U, dt)
    X_arr = np.append(X_arr, X, axis=1) # append X : plots

# print pitch final value
print(f'pitch final value : {X_arr[0, -1]} rad')

# plot
plt.subplot(4, 1, 1)
plt.title('Coded PID with saturation with FTBO sim (custom impl of integration)')
plt.plot(t_pid, X_arr[0, :], 'blue')
plt.plot(t_pid, X_ref[0,0] * np.ones(t_pid.shape), 'red')
plt.legend(labels=('pitch', 'pitch_ref'))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t_pid, X_arr[1, :], 'blue')
plt.legend(labels=('pitch_dot',))
plt.ylabel('[rad/s]')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t_pid[0:tsteps], U_arr[1, 1:], 'green')
plt.legend(labels=('elevator',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(t_pid, err_arr, 'black')
plt.legend(labels=('error',))
plt.ylabel('[rad]')
plt.grid()

plt.show()
