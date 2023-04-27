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
# roll
# mat A : dynamique du roll
A_roll: np.ndarray = np.array([[0, 1            ],
                               [0, -uav.a_roll1]])

# mat B : commande du pitch
B_roll: np.ndarray = np.array([[0, 0           ],
                               [0, uav.a_roll2]])

# O and cmd_de = 0 (elevator deflection) for initial command
cmd_da: float = 0.0
U_roll: np.ndarray = np.array([[0],
                         [cmd_da]])

# vecteur d'état : pitch et pitch_dot (0,0) pour la valeur initiale
X_roll: np.ndarray = np.array([[0],
                          [0]])

# vecteur dérivée de l'état : pitch_dot et pitch_dot_dot (0,0) pour la valeur initiale
X_roll_dot: np.ndarray = np.array([[0],
                              [0]])

# course
A_course: float = 0

# course angle : 0 at init
X_course: float = 0

# B matrix
B_course: float = uav.G / uav.Va_trim

# roll command
U_course: float = 0

# timestep (120Hz)
dt: float = 1/120

# reference
X_course_ref: float = 45.0 * (np.pi / 180) # deg to rad

# placeholder ref, will be filled by the h_pid
X_roll_ref: np.ndarray = np.array([[0        ],
                                    [0        ]])

# arrays for plotting, initializing them with the initial value of X, U and X_ref - X
# pitch
X_roll_arr: np.array = np.array(X_roll)
U_roll_arr: np.array = np.array(U_roll)
err_roll_arr: np.array = np.array([X_roll_ref[0,0] - X_roll[0,0]])

# altitude
X_course_arr: np.array = np.array(X_course)
U_course_arr: np.array = np.array(U_course)
err_course_arr: np.array = np.array(X_course_ref - X_course)

# PID gains
K_lat: dict[str, float]
K_lat, _= uav.compute_lat_pid_gains()
kp_roll: float = K_lat['kp_roll']
ki_roll: float = K_lat['ki_roll']
kd_roll: float = K_lat['kd_roll']
kp_course: float = K_lat['kp_course']
ki_course: float = K_lat['ki_course']
roll_pid: PID = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll, dt=dt, limit=uav.aileron_limit, is_throttle=False)
course_pid: PID = PID(kp=kp_course, ki=ki_course, kd=0, dt=dt, limit=uav.roll_max, is_throttle=False)

tsteps: int = 10*120
t_pid: np.ndarray = np.linspace(0, 10, tsteps+1) # +1 car on a une val initiale dans X_arr au début (plotting)
for i in range (0, tsteps):
    err_pitch: float
    err_h: float

    # set reference altitude to attain
    course_pid.set_reference(X_course_ref)
    U_course, err_course = course_pid.update(state=X_course, saturate=True) # computing the according pitch command
    U_course_arr = np.append(U_course_arr, U_course)
    err_course_arr  = np.append(err_course_arr, err_course)

    roll_pid.set_reference(U_course) # set reference
    U_roll[1], err_roll = roll_pid.update(state=X_roll[0,0], state_dot=X_roll[1,0], saturate=True) # get command and error
    U_roll_arr = np.append(U_roll_arr, U_roll, axis=1) # append U : plots
    err_roll_arr = np.append(err_roll_arr, err_roll) # append error : plots

    # integrate the model and get the next X state
    X_roll = integrate_ss_mat(A_roll, B_roll, X_roll, U_roll, dt)
    X_roll_arr = np.append(X_roll_arr, X_roll, axis=1) # append X : plots

    X_course = integrate_ss(A_course, B_course, X_course, X_roll[0,0], dt) # integrating the altitude model
    X_course_arr = np.append(X_course_arr, X_course)

# plot
plt.subplot(4, 2, 1)
plt.title('Inner loop : roll -> aileron cmd')
plt.plot(t_pid, X_roll_arr[0, :], 'blue')
plt.plot(t_pid[0:tsteps], U_course_arr[1:], 'red')
plt.legend(labels=('roll', 'roll_ref'))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(t_pid, X_roll_arr[1, :], 'blue')
plt.legend(labels=('roll_dot',))
plt.ylabel('[rad/s]')
plt.grid()

plt.subplot(4, 2, 5)
plt.plot(t_pid[0:tsteps], U_roll_arr[1, 1:], 'green')
plt.legend(labels=('aileron_cmd',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 7)
plt.plot(t_pid, err_roll_arr, 'black')
plt.legend(labels=('error_roll',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 2)
plt.title('Outer loop : course -> roll cmd')
plt.plot(t_pid, X_course_arr, 'blue')
plt.plot(t_pid, X_course_ref * np.ones(t_pid.shape), 'red')
plt.legend(labels=('course', 'course_ref'))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 4)
plt.plot(t_pid[0:tsteps], U_course_arr[1:], 'green')
plt.legend(labels=('roll_cmd',))
plt.ylabel('[rad]')
plt.grid()

plt.subplot(4, 2, 6)
plt.plot(t_pid, err_course_arr, 'black')
plt.legend(labels=('error_course',))
plt.ylabel('[rad]')
plt.grid()


plt.show()