import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/../..')
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint
from agents.pid import PID
from scipy import linalg

# integrate the state space model (matrix form)
def integrate_ss_mat(A: np.ndarray, B: np.ndarray, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    X_out: np.ndarray = linalg.expm(A * dt) @ X + dt * linalg.expm(A * 0) @ B @ U
    return X_out

# integrate the state space model (scalar form, case where dim(A)=dim(B)=dim(X)=dim(U)=1)
def integrate_ss(A, B, X, U, dt) -> float:
    X_out: float = np.exp(A * dt) * X + dt * np.exp(A * 0) * B * U
    return X_out

trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# matrices A et B
# pitch
# mat A : dynamique du pitch
A_pitch: np.ndarray = np.array([[0            , 1            ],
                          [-uav.a_pitch2, -uav.a_pitch1]])

# mat B : commande du pitch
B_pitch: np.ndarray = np.array([[0, 0           ],
                          [0, uav.a_pitch3]])

# O and cmd_de = 0 (elevator deflection) for initial command
cmd_de: float = 0.0
U_pitch: np.ndarray = np.array([[0],
                         [cmd_de]])

# vecteur d'état : pitch et pitch_dot (0,0) pour la valeur initiale
X_pitch: np.ndarray = np.array([[0],
                          [0]])

# vecteur dérivée de l'état : pitch_dot et pitch_dot_dot (0,0) pour la valeur initiale
X_pitch_dot: np.ndarray = np.array([[0],
                              [0]])

# altitude (h)
A_h: float = 0

X_h: float = 0

B_h: float = uav.Va_trim

U_h: float = 0

# timestep (120Hz)
dt: float = 1/120

# reference : pitch_ref = une ref et pitch_dot_ref = 0 -> on veut vitesse nulle à l'arrivée
# pitch_ref: float = 45.0 * (np.pi / 180) # deg to rad
X_h_ref: float = 1
# pitch_ref: float = 1.0 # for step response testing

# placeholder ref, will be filled by the h_pid
X_pitch_ref: np.ndarray = np.array([[0        ],
                                    [0        ]])

# arrays for plotting, initializing them with the initial value of X, U and X_ref - X
# pitch
X_pitch_arr: np.array = np.array(X_pitch)
U_pitch_arr: np.array = np.array(U_pitch)
err_pitch_arr: np.array = np.array([X_pitch_ref[0,0] - X_pitch[0,0]])

# altitude
X_h_arr: np.array = np.array(X_h)
U_h_arr: np.array = np.array(U_h)
err_h_arr: np.array = np.array(X_h_ref - X_h)

# PID gains
K_long: dict[str, float]
K_long, _, __ = uav.compute_long_pid_gains()
kp_pitch: float = K_long['kp_pitch']
kd_pitch: float = K_long['kd_pitch']
kp_h: float = K_long['kp_h']
ki_h: float = K_long['ki_h']
pitch_pid: PID = PID(kp=kp_pitch, ki=0, kd=kd_pitch, dt=dt, limit=uav.elevator_limit, is_throttle=False)
h_pid: PID = PID(kp=kp_h, ki=ki_h, kd=0, dt=dt, is_throttle=False)

tsteps: int = 10*120
t_pid: np.ndarray = np.linspace(0, 10, tsteps+1) # +1 car on a une val initiale dans X_arr au début (plotting)
for i in range (0, tsteps):
    err_pitch: float
    err_h: float

    # set reference altitude to attain
    h_pid.set_reference(X_h_ref)
    U_h, err_h = h_pid.update(state=X_h, saturate=False) # computing the according pitch command
    U_h_arr = np.append(U_h_arr, U_h)
    err_h_arr  = np.append(err_h_arr, err_h)

    pitch_pid.set_reference(U_h) # set reference
    U_pitch[1], err_pitch = pitch_pid.update(state=X_pitch[0,0], state_dot=X_pitch[1,0], saturate=True) # get command and error
    U_pitch_arr = np.append(U_pitch_arr, U_pitch, axis=1) # append U : plots
    err_pitch_arr = np.append(err_pitch_arr, err_pitch) # append error : plots

    # integrate the model and get the next X state
    X_pitch = integrate_ss_mat(A_pitch, B_pitch, X_pitch, U_pitch, dt)
    X_pitch_arr = np.append(X_pitch_arr, X_pitch, axis=1) # append X : plots

    X_h = integrate_ss(A_h, B_h, X_h, X_pitch[0,0], dt) # integrating the altitude model
    X_h_arr = np.append(X_h_arr, X_h)

# plot
plt.subplot(4, 2, 1)
plt.title('Inner loop : pitch -> elevator cmd')
plt.plot(t_pid, X_pitch_arr[0, :], 'blue')
plt.plot(t_pid[0:tsteps], U_h_arr[1:], 'red')
plt.legend(labels=('pitch', 'pitch_ref'))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(t_pid, X_pitch_arr[1, :], 'blue')
plt.legend(labels=('pitch_dot',))
plt.ylabel('[rad/s]')
plt.grid()

plt.subplot(4, 2, 5)
plt.plot(t_pid[0:tsteps], U_pitch_arr[1, 1:], 'green')
plt.legend(labels=('elevator_cmd',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 7)
plt.plot(t_pid, err_pitch_arr, 'black')
plt.legend(labels=('error_pitch',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 2)
plt.title('Outer loop : altitude -> pitch cmd')
plt.plot(t_pid, X_h_arr, 'blue')
plt.plot(t_pid, X_h_ref * np.ones(t_pid.shape), 'red')
plt.legend(labels=('altitude', 'altitude_ref'))
plt.ylabel('[m]')
plt.grid()

plt.subplot(4, 2, 4)
plt.plot(t_pid[0:tsteps], U_h_arr[1:], 'green')
plt.legend(labels=('pitch_cmd',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 6)
plt.plot(t_pid, err_h_arr, 'black')
plt.legend(labels=('error_alt',))
plt.ylabel('[m]')
plt.grid()


plt.show()
