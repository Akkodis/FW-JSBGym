"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import numpy as np
import jsbsim
from scipy.optimize import minimize
from scipy.spatial.transform.rotation import Rotation as R
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


def CX_coeffs(alpha: float, mav: AeroModel) -> tuple[float, float, float]:
    CX_alpha: float = -CD(alpha, mav) * np.cos(alpha) + CL(alpha, mav) * np.sin(alpha)
    CXq_alpha: float = -mav.CDq * np.cos(alpha) + mav.CLq * np.sin(alpha)
    CXde_alpha: float = -mav.CDde * np.cos(alpha) + mav.CLde * np.sin(alpha)

    return CX_alpha, CXq_alpha, CXde_alpha


def CZ_coeffs(alpha: float, mav: AeroModel) -> tuple[float, float, float]:
    CZ_alpha: float = -CD(alpha, mav) * np.sin(alpha) - CL(alpha, mav) * np.cos(alpha)
    CZq_alpha: float = -mav.CDq * np.sin(alpha) - mav.CLq * np.cos(alpha)
    CZde_alpha: float = -mav.CDde * np.sin(alpha) - mav.CLde * np.cos(alpha)

    return CZ_alpha, CZq_alpha, CZde_alpha

def trimmed_state(mav: AeroModel, Va, alpha, beta) -> np.ndarray:
    u_star: float = Va*np.cos(alpha) + Va*np.cos(beta)
    v_star: float = Va*np.sin(beta)
    w_star: float = Va*np.sin(alpha)+ Va*np.cos(beta)
    theta_star = alpha + beta
    p_star: float = 0.0 # since R=inf (straight flight), p=q=r=0
    q_star: float = 0.0
    r_star: float = 0.0

    return np.array([[u_star], [v_star], [w_star], [theta_star], [p_star], [q_star], [r_star]])

def trimmed_input(mav: AeroModel, Va, alpha, beta, u, v, w, phi, theta, psi, p, q, r) -> np.ndarray:
    # elevator de
    # intermidiate variable de_A
    de_A: float = ((mav.Ixz * (p**2 - r**2) + (mav.Ixx - mav.Izz) * p * r) / (1/2 * mav.rho * Va**2 * mav.c * mav.S))
    de_star: float = (de_A - mav.Cmo - mav.Cma * alpha - mav.Cmq * (mav.c * q / (2 * Va))) / mav.Cmde

    # throttle dt
    CX_alpha: float
    CXq_alpha: float
    CXde_alpha: float
    CX_alpha, CXq_alpha, CXde_alpha = CX_coeffs(alpha, mav)
    dt_star: float = mav.Khp2w/(mav.Khp2ftlbsec * mav.Pwatt) * (-r*v + q*w + mav.G*np.sin(theta) \
                    - ((mav.rho * Va**2 * mav.S)/(2 * mav.mass)) * \
                    (CX_alpha + CXq_alpha * ((mav.c * q) / (2*Va)) + CXde_alpha * de_star))

    # aileron da. Since no rudder, pick da has 2 equal expressions. I chose the one with 1/Cpda
    # intermidiate variable da_A
    da_A: float = (mav.gamma1*p*q + mav.gamma2*p*q) / (1/2 * mav.rho * Va**2 * mav.S * mav.b)
    da_star: float = 1/mav.Cpda * (-da_A - mav.Cpo - mav.Cpbeta * beta - mav.Cpp * ((mav.b*p)/(2*Va)) - mav.Cpr * ((mav.b*r)/(2*Va)))

    # rudder = 0 since no rudder on x8
    dr_star: float = 0
    return np.array([de_star, da_star, dr_star, dt_star])


def compute_trim(mav, Va, gamma):
    # define initial state and input

    ##### TODO #####
    # set the initial conditions of the optimization
    h: float = 600 # altitude m
    alpha0: float = 0
    beta0: float = 0
    phi0: float = 0
    psi0: float = 0
    
    trim_state0: np.ndarray = trimmed_state(mav, Va, alpha0, beta0)
    print(trim_state0)
    theta0: float = trim_state0[3]
    e0: np.ndarray = Euler2Quaternion(0, theta0, 0)
    state0: np.ndarray = np.array([
                   [0], # 0 pn
                   [0],  # 1 pe
                   [h],  # 2 pd, h
                   [trim_state0[0]],  # 3 u, alpha = gamma (no wind) and beta = 0
                   [trim_state0[1]], # 4 v, beta = 0 for having force side velocity to be zero
                   [trim_state0[2]], # 5 w
                   [e0[0]],  # 6 e0, previously set to 1
                   [e0[1]],  # 7 e1
                   [e0[2]],  # 8 e2
                   [e0[3]],  # 9 e3
                   [trim_state0[4]], # 10 p = 0. Since R=inf p=q=r=0
                   [trim_state0[5]], # 11 q = 0
                   [trim_state0[6]]  # 12 r = 0
                   ])

    trim_input0 = trimmed_input(mav, Va, alpha0, beta0, state0[3], state0[4], state0[5], phi0, theta0, psi0, state0[10], state0[11], state0[12])
    delta0 = np.array([[trim_input0[0]],  # 13 elevator
                       [trim_input0[1]],  # 14 aileron
                       [trim_input0[2]],  # 15 rudder = 0
                       [trim_input0[3]]]) # 16 throttle

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
    trim_state: np.ndarray = np.array([res.x[0:13]]).T
    trim_input: np.ndarray = np.array([res.x[13:17]]).T

    print('trim_state=', trim_state.T)
    print('trim_input=', trim_input.T)
    return trim_state, trim_input

def compute_f_xu(mav: AeroModel, Va, alpha, beta, u, v, w, phi, theta, psi, p, q, r, de, da, dr, dt):
    # pn_dot
    # pn_dot: float = (np.cos(theta) * np.cos(psi)) * u \
    #                 + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v \
    #                 + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w
    pn_dot: float = 0.0

    # pe_dot
    # pe_dot: float = (np.cos(theta) * np.sin(psi)) * u \
    #                 + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v \
    #                 + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w
    pe_dot: float = 0.0
    
    # h_dot
    h_dot: float = u * np.sin(theta) - v * np.sin(phi) * np.cos(theta) - w * np.cos(phi) * np.cos(theta)

    # u_dot
    CX_alpha: float
    CXq_alpha: float
    CXde_alpha: float
    CX_alpha, CXq_alpha, CXde_alpha = CX_coeffs(alpha, mav)
    u_dot: float = r*v - q*w - mav.G*np.sin(theta) \
                    + ((mav.rho * Va**2 * mav.S) / (2*mav.mass)) * (CX_alpha + CXq_alpha * ((mav.c*q) / (2*Va)) + CXde_alpha * de) \
                    + (((mav.Pwatt * dt) / mav.Khp2w) * mav.Khp2ftlbsec)

    # v_dot
    v_dot: float = p*w - r*u + mav.G*np.cos(theta)*np.sin(phi) \
                    + ((mav.rho * Va**2 * mav.S)/(2*mav.mass)) \
                    * (mav.CYo + mav.CYb*beta + mav.CYp * ((mav.b*p)/(2*Va)) \
                    + mav.CYr * ((mav.b*r)/(2*Va)) + mav.CYda*da + mav.CYdr*dr)
    # w_dot
    CZ_alpha: float
    CZq_alpha: float
    CZde_alpha: float
    CZ_alpha, CZq_alpha, CZde_alpha = CZ_coeffs(alpha, mav)
    w_dot: float = q*u - p*v + mav.G*np.cos(theta)*np.cos(phi) \
                    + ((mav.rho * Va**2 * mav.S)/(2*mav.mass)) \
                    * (CZ_alpha + CZq_alpha * ((mav.c*q)/(2*Va)) + CZde_alpha * de)
    
    # phi_dot
    phi_dot: float = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)

    # theta_dot
    theta_dot: float = q*np.cos(phi) - r*np.sin(phi)

    # psi_dot
    psi_dot: float = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

    # p_dot
    p_dot: float = mav.gamma1*p*q - mav.gamma2*q*r + (1/2*mav.rho*Va**2*mav.S*mav.b) \
                    * (mav.Cpo + mav.Cpbeta*beta +mav.Cpp*((mav.b*p)/(2*Va)) + mav.Cpda*da + mav.Cpdr*dr)
    
    # q_dot
    q_dot: float = mav.gamma5*p*r - mav.gamma6*(p**2 - r**2) + ((mav.rho * Va**2 * mav.S * mav.c)/(2*mav.Iy)) \
                    * (mav.Cmo + mav.Cma*alpha + mav.Cmq*((mav.c*q)/(2*Va)) + mav.Cmde*de)
    
    # r_dot
    r_dot: float = mav.gamma7*p*q - mav.gamma1*q*r + (1/2*mav.rho*Va**2*mav.S*mav.b) \
                    * (mav.Cro + mav.Crbeta*beta + mav.Crp*((mav.b*r)/(2*Va)) + mav.Crr*((mav.b*r)/(2*Va)) + mav.Crda*da + mav.Crdr*dr)

    return np.array([[pn_dot], [pe_dot], [h_dot], [u_dot], [v_dot], [w_dot], [phi_dot], [theta_dot], [psi_dot], [p_dot], [q_dot], [r_dot]]) # state_dot


def trim_objective_fun(x, mav, Va, gamma) -> float:
    # objective function to be minimized
    ##### TODO #####
    J: float = 0.0

    # compute state_dot_star (Eq 5.21)
    e_dot_star = Euler2Quaternion(0, 0, 0) # psi = 0 because Va/R with R=inf is 0
    state_dot_star = np.array([[0], # pn_dot_star
                                [0], # pe_dot_star
                                [Va * np.sin(gamma)], # pd_dot_star or h_dot_star (=0 since gamma=0)
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

    # TODO: ASK : What value of alpha to use?
    attitude: R = R.from_quat([x[6], x[7], x[8], x[9]])
    attitude.as_euler('xyz')
    alpha: float = attitude[1] - gamma # alpha = theta - gamma
    beta: float = 0.0 # for having no sideforce -> v=0
    
    # compute trimmed states and input
    state_star: np.ndarray = trimmed_state(mav, Va, alpha, beta)
    input_star: np.ndarray = trimmed_input(mav, Va, alpha, beta, x[3], x[4], x[5], attitude[0], attitude[1], attitude[2], x[10], x[11], x[12])

    # compute f(state_star, input_star)
    f_xu: np.ndarray= compute_f_xu(mav, Va, alpha, beta, state_star[0], state_star[1], state_star[2], 0, state_star[3], 0, \
                                    state_star[4], state_star[5], state_star[6], input_star[0], input_star[1], input_star[2],\
                                    input_star[3])

    # compute the objective function
    J = np.linalg.norm(state_dot_star - f_xu)
    return J

def main():
    Va: float
    gamma: float

    fdm: jsbsim.FGFDMExec = jsbsim.FGFDMExec(None)
    fdm.load_model("x8")
    ic_path = 'initial_conditions/x8_basic_ic.xml'
    fdm.load_ic(ic_path, False)
    fdm.run_ic()
    mav: AeroModel = AeroModel(fdm)

    for Va in range(8, 23): # trying Va for 8 to 23 m/s -> 28 to 83 km/h
        for gamma in range(-10, 10):
            gamma = np.deg2rad(gamma)
            res = compute_trim(mav, Va, gamma)

if __name__ == "__main__":
    main()