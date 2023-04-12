import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import control as ctl
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint
from agents.pid import PID
from scipy import linalg


def integrate_ss(A: np.ndarray, B: np.ndarray, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    X_out: np.ndarray = linalg.expm(A * dt) @ X + dt * linalg.expm(A * 0) @ B @ U
    return X_out

trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# matrices A et B
# mat A : dynamique du roll
A: np.ndarray = np.array([[0, 0, 1           ],
                         [1, 0, 0           ],
                         [0, 0, -uav.a_roll1]])

B: np.ndarray = np.array([[0, 0, 0          ],
                         [0, 0, 0          ],
                         [0, 0, uav.a_roll2]])

cmd_da: float = 0.0
U: np.ndarray = np.array([[0],
                          [0],
                          [cmd_da]])

X: np.ndarray = np.array([[0],
                          [0],
                          [0]])

X_dot: np.ndarray = np.array([[0],
                              [0],
                              [0]])

roll_ref: float = 45.0 * (np.pi / 180) # deg to rad
X_ref: np.ndarray = np.array([[roll_ref],
                              [0],
                              [0]])

dt: float = 1/120
X_arr: np.array = np.array(X)
U_arr: np.array = np.array(U)
err_arr: np.array = np.array([X_ref[0,0] - X[0,0]])

# PID Gains
K_lat: dict[str, float]
K_lat, _= uav.compute_lat_pid_gains()
kp_roll: float = K_lat['kp_roll']
ki_roll: float = K_lat['ki_roll']
kd_roll: float = K_lat['kd_roll']

roll_pid: PID = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll, dt=dt, limit=uav.aileron_limit, is_throttle=False)
tsteps: int = 10*120
t_pid: np.array = np.linspace(0, 10, tsteps+1)

for i in range (0, tsteps):
    err: float
    roll_pid.set_reference(X_ref[0,0]) # set reference
    U[1], err = roll_pid.update(state=X[0,0], state_dot=X[2,0], saturate=True) # get command and error
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
