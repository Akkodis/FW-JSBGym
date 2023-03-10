"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from message_types.msg_delta import MsgDelta
import time
from models.aerodynamics import AeroModel

def Euler2Quaternion(phi, theta, psi) -> np.ndarray:
    """
    Converts an euler angle attitude to a quaternian attitude
    :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
    :return: Quaternian attitude in np.array(e0, e1, e2, e3)
    """

    e0: float = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1: float = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2: float = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3: float = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.array([[e0],[e1],[e2],[e3]])

def CL(alpha: float, mav: AeroModel) -> float:
    return mav.CLo + mav.CLalpha * alpha # linear approximation of lift coefficient

def CD(alpha: float, mav: AeroModel) -> float:
    return mav.CDo + mav.CDalpha * alpha # Linear approximation of drag coefficient

def compute_trimmed_input(mav: AeroModel, Va, alpha, beta, u, v, w, phi, theta, psi, p, q, r):
    # elevator
    de_0: float = ((mav.Ixz * (p**2 - r**2) + (mav.Ixx - mav.Izz) * p * r) / (1/2 * mav.rho * Va**2 * mav.c * mav.S))
    de_star: float = (de_0 - mav.Cmo - mav.Cma * alpha - mav.Cmq * (mav.c * q / (2 * Va))) / mav.Cmde

    # throttle
    CX_alpha: float = -CD(alpha, mav) * np.cos(alpha) + CL(alpha, mav) * np.sin(alpha)
    CXq_alpha: float = -mav.CDq * np.cos(alpha) + mav.CLq * np.sin(alpha)
    CXde_alpha: float = -mav.CDde * np.cos(alpha) + mav.CLde * np.sin(alpha)
    dt_0: float = 2 * mav.mass * (-r * v + q * w + mav.G * np.sin(theta)) - mav.rho * Va**2 * mav.S * (CX_alpha + CXq_alpha * (mav.c * q / (2 * Va) + CXde_alpha * de_star))
    dt_star: float = (dt_0 / (mav.rho * mav.Sprop * mav.Cprop * mav.k_motor**2)) + (Va**2 / mav.k_motor**2)

    # TODO: aileron

    # rudder = 0 since no rudder on x8
    dr_star: float = 0
    pass

def compute_trim(mav, Va, gamma):
    # define initial state and input

    ##### TODO #####
    # set the initial conditions of the optimization
    # e0: np.ndarray = Euler2Quaternion(0., gamma, 0.)
    h: float = 400 # altitude m
    alpha: float = 0
    theta: float = alpha + gamma
    beta: float = 0
    phi: float = 0
    e0: np.ndarray = Euler2Quaternion(0, theta, 0)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [h],  # pd, h
                   [Va*np.cos(alpha) + Va*np.cos(beta)],  # u, alpha = gamma (no wind) and beta = 0
                   [Va*np.sin(beta)], # v, beta = 0 for having force side velocity to be zero
                   [Va*np.sin(alpha)+ Va*np.cos(beta)], # w
                   [e0[0]],  # e0, previously set to 1
                   [e0[1]],  # e1
                   [e0[2]],  # e2
                   [e0[3]],  # e3
                   [0], # p
                   [0], # q
                   [0]  # r
                   ])
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized
    ##### TODO #####
    J = 0

    # compute state_dot_star (Eq 5.21)
    e_dot_star = Euler2Quaternion(0, 0, (Va/mav) * np.cos(gamma))
    state_dot_star = np.array([[0], #pn_dot_star
                                [0], # pe_dot_star
                                [Va * np.sin(gamma)], # pd_dot_star, h_dot_star
                                [0], # u_dot_star
                                [0], # v_dot_star
                                [0], # w_dot_star
                                [e_dot_star[0]], # e0_dot_star
                                [e_dot_star[1]], # e1_dot_star
                                [e_dot_star[2]], # e2_dot_star
                                [e_dot_star[3]], # e3_dot_star
                                [0], # p_dot_star
                                [0], # q_dot_star
                                [0]  # r_dot_star
                            ])


    
    return J