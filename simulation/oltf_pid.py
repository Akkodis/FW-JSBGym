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

# t = np.linspace(0, 0.5, 101)

# # # step response
# # # _, y0 = control.step_response(H, t, input=0, squeeze=True)

# # forced response
# dt_: np.ndarray = 0.2 * np.ones(t.shape)
# theta_: np.ndarray = np.zeros(t.shape)
# dv: np.ndarray = np.zeros(t.shape)
# u: np.ndarray = np.array([dt_, theta_, dv]) # commands : steps
# _, y = control.forced_response(H, t, U=u, squeeze=True)
# # print last output

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

t: np.ndarray = np.linspace(0, 0.005, 100) # temps de simulation de 0.5s, 100 points au total
Va_ref: float = 0.2 # set reference as deviation from Va_trim m/s
Va_: float = 0.0 # initial value for Va_
Va_array: list = [Va_]
cmds_th_: list = []
errs: list = []

# compute pid gains airspeed hold with commanded throttle v_th
K_vth: dict[str, float] = uav.compute_long_pid_gains()
vth_pid: PID = PID(kp=K_vth['kp_vth'], ki=K_vth['ki_vth'], kd=0, dt=1/120, limit=uav.throttle_limit, is_throttle=True)

tsteps: int = 15*120
t_pid: np.ndarray = np.linspace(0, 15, tsteps+1) # temps de simulation de 15s, 15*120 points au total (+1 à cause de la list Va_, qui n'est pas vide au début)
for i in range(0, 15*120):
    vth_pid.set_reference(Va_ref) # je set une reference ici Va_ref = 0.2
    cmd_th_, err = vth_pid.update(Va_, normalize=False) # update du PID, me retourne une commande et l'erreur. initialisation Va_ = 0.0
    cmds_th_.append(cmd_th_) # je remplis des list pour plot
    errs.append(err)

    # les commandes, il y en a 3
    dt_: np.ndarray = cmd_th_ * np.ones(t.shape) # je set la commande cmd_th_ donnée par le PID, appelée dt_ dans la TF du bouquin
    theta_: np.ndarray = np.zeros(t.shape) # je set une commande theta_ nulle, vu que je veux tester uniquement le maintien de l'airspeed via une commande en throttle
    dv: np.ndarray = np.zeros(t.shape) # un terme de perturbation dv géré comme une commande dans la TF du bouquin, nul dans le cas simple
    u: np.ndarray = np.array([dt_, theta_, dv]) # la commande u est la concaténation des trois commandes dt_, theta_ et dv

    # je simule la réponse sur un certain temps t
    # le temps t est de 0 à 0.5s, 100 points au total (je sais pas si c'est assez court)
    # l'argument X0 est la condition initiale, je la set à Va_
    _, Va_t = control.forced_response(sys=H, T=t, X0=Va_, U=u, squeeze=True)

    # je prends la première valeur de la réponse Va_(t), et je la stocke dans Va_
    Va_ = Va_t[-1]
    Va_array.append(Va_) # je remplis la list Va_array pour plot


# plotting
plt.close('all')
fig_width_cm = 24
fig_height_cm = 18
# plt.figure(1 , figsize =(fig_width_cm /2.54 , fig_height_cm /2.54))
plt.subplot(3 , 1 , 1)
plt.plot(t_pid, Va_array ,'blue')
plt.plot(t_pid , Va_ref * np.ones(t_pid.shape) ,'red')
plt.grid()
plt.legend(labels =('Va_', 'Va_ref'))

plt.subplot(3 , 1 , 2)
plt.plot(t_pid[0:tsteps] , cmds_th_ ,'green')
# plt.xlabel('t [ s ]')
plt.grid()
plt.legend(labels =('cmd_th_',))

plt.subplot(3 , 1 , 3)
plt.plot(t_pid[0:tsteps] , errs ,'black')
plt.xlabel('t [ s ]')
plt.grid()
plt.legend(labels =('error',))


plt.show()