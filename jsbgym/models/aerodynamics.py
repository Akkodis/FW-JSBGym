import math
import numpy as np
from jsbgym.trim.trim_point import TrimPoint

class AeroModel(object):
    G: float = 9.82  # m/s2

    def __init__(self, trim: TrimPoint = TrimPoint("x8")):

        # trim point
        self.trim: TrimPoint = trim

        # Inertia matrix coefficients
        self.Ixx: float = 1.2290  # kg*m2
        self.Iyy: float = 0.1702
        self.Izz: float = 0.8808
        self.Ixz: float = 0.9343
        self.Ixy: float = 0.0
        self.Iyz: float = 0.0

        # Air density
        self.rho: float = 1.155992485886005

        # Mass of the aircraft
        self.mass: float = 3.3640  # kg

        # gamma coefficients
        self.gamma: float = self.Ixx * self.Izz - self.Ixz ** 2
        self.gamma1: float = (self.Ixz * (self.Ixx - self.Iyy + self.Izz)) / self.gamma
        self.gamma2: float = (self.Izz * (self.Izz - self.Iyy) + self.Ixz ** 2) / self.gamma
        self.gamma3: float = self.Izz / self.gamma
        self.gamma4: float = self.Ixz / self.gamma
        self.gamma5: float = (self.Izz - self.Ixx) / self.Iyy
        self.gamma6: float = self.Ixz / self.Iyy
        self.gamma7: float = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz ** 2) / self.gamma
        self.gamma8: float = self.Ixx / self.gamma

        # Geometric parameters of the aircraft
        self.Va_trim: float = self.trim.Va_ms   # trim airspeed, chosen to be 33kts -> 16.97 m/s or 61.12 km/h : include it here ?
        self.S: float = 0.75  # wing span m2
        self.b: float = 2.10  # wing span m
        self.c: float = 0.3571 # mean aerodynamic chord m

        # Propeller parameters according to Small Unmanned Aircrafts Book
        # the jsb simulated x8 doesn't use this, keeping it here for reference
        self.Sprop: float = 0.3048 * math.pi # propeller area m2 diam*PI
        self.Cprop: float = 1.0  # propeller coefficient
        self.k_motor: float = 80.0  # motor constant

        # Direct Thruster Parameters:
        self.Khp2w: float = 745.7  # hp to watts constant
        self.Khp2ftlbsec: float = 550.0  # hp to ft-lb/sec constant
        self.Pwatt: float = 8.0  # power in watts

        # Aero coeffs of the x8
        # Lift
        self.CLo: float = 0.0867
        self.CLalpha: float = 4.0203
        self.CLq: float = 3.87
        self.CLde: float = 0.2781

        # Drag
        self.CDo: float = 0.0197
        self.CDalpha: float = 0.0791
        self.CDq: float = 0.0000
        self.CDde: float = 0.0633

        # Sideforce: Y
        self.CYo: float = 0.0
        self.CYb: float = -0.2239
        self.CYp: float = -0.1374
        self.CYr: float = 0.0839
        self.CYda: float = 0.0433
        self.CYdr: float = 0.0 # no rudder -> equals 0.0

        # roll moment : l
        self.Clo: float = 0.0
        self.Clp: float = -0.4042
        self.Clb: float = -0.0849
        self.Clr: float = 0.0555
        self.Clda: float = 0.1202
        self.Cldr: float = 0.0 # no rudder -> equals 0.0

        # pitch moment : m
        self.Cmq: float = -1.3012
        self.Cma: float = -0.4629
        self.Cmde: float = -0.2292
        self.Cmo: float = 0.0227

        # yaw moment : n
        self.Cno: float = 0.0
        self.Cnp: float = 0.0044
        self.Cnb: float = 0.0283
        self.Cnr: float = -0.072
        self.Cnda: float = 0.1202
        self.Cndr: float = 0.0

        # intermediate values of aero coeffs weighted by inertia components
        self.Cpp: float = self.gamma3 * self.Clp + self.gamma4 * self.Cnp
        self.Cpda: float = self.gamma3 * self.Clda + self.gamma4 * self.Cnda
        self.Cpo: float = self.gamma3 * self.Clo + self.gamma4 * self.Cno # Clo and Cno are 0.0 because x8 UAV is symetric -> Cpo = 0.0
        self.Cpbeta: float = self.gamma3 * self.Clb + self.gamma4 * self.Cnb
        self.Cpr: float = self.gamma3 * self.Clr + self.gamma4 * self.Cnr
        self.Cpdr: float = self.gamma3 * self.Cldr + self.gamma4 * self.Cndr # no rudder -> equals 0.0

        self.Cro: float = self.gamma4 * self.Clo + self.gamma8 * self.Cno # Clo and Cno are 0.0 because x8 UAV is symetric -> Cro = 0.0
        self.Crbeta: float = self.gamma4 * self.Clb + self.gamma8 * self.Cnb
        self.Crp: float = self.gamma4 * self.Clp + self.gamma8 * self.Cnp
        self.Crr: float = self.gamma4 * self.Clr + self.gamma8 * self.Cnr
        self.Crda: float = self.gamma4 * self.Clda + self.gamma8 * self.Cnda
        self.Crdr: float = self.gamma4 * self.Cldr + self.gamma8 * self.Cndr # no rudder -> equals 0.0


        # lateral TF coeffs
        # roll
        self.a_roll1: float = -1 / 2 * self.rho * self.Va_trim ** 2 * self.S * self.b * self.Cpp * (
                        self.b / (2 * self.Va_trim))
        self.a_roll2: float = 1 / 2 * self.rho * self.Va_trim ** 2 * self.S * self.b * self.Cpda

        self.aileron_limit: float = 1.04 # combined aileron actuator max deflection : rad
        self.roll_max: float = 45.0 * (math.pi / 180)  # roll max angle : deg to rad
        self.roll_err_max: float = self.roll_max * 2  # max expected error, roll_max * 2 : rad
        self.roll_damping: float = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink
        self.course_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink


        # longitutinal TF coefficients
        # pitch
        self.a_pitch1: float = - ((self.rho * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cmq * (self.c / (2 * self.Va_trim))
        self.a_pitch2: float = - ((self.rho * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cma
        self.a_pitch3: float = ((self.rho * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cmde
        self.elevator_limit: float = 30.0 * (math.pi / 180)  # elevator actuator max deflection : deg to rad
        self.pitch_max: float = 15.0 * (math.pi / 180)  # pitch max angle : deg to rad
        self.pitch_err_max: float = self.pitch_max * 2  # max expected error, pitch_max * 2 : rad
        self.pitch_damping: float = 1  # ask if needed to plot the step responses for various damping ratios in something
        self.h_damping: float = 1 # same as above but for altitude

        # Airspeed hold using throttle
        self.throttle_limit: float = 1.0  # throttle actuator max value
        self.av1: float = ((self.rho * trim.Va_ms * self.S) / self.mass) / (self.CDo + self.CDalpha * trim.alpha_rad + self.CDde * trim.elevator)
        self.av2: float = (self.Pwatt / self.Khp2w) * self.Khp2ftlbsec
        self.av3: float = self.G * np.cos(trim.theta_rad - trim.alpha_rad)
        self.v_damping: float = 1
        self.v_puls: float = 2

        # self.compute_lat_pid_gains()
        # self.compute_long_pid_gains()


    def compute_lat_pid_gains(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Computes the PID gains for the lateral control system
        :return: a tuple of two dictionaries, the first one containing the gains for the roll angle attitude control (inner loop) +
        course angle hold with commanded roll angle (outer loop), the second one containing the response times of each loop
        """
        # PID gains for roll attitude control (inner loop)
        kp_roll: float = self.aileron_limit / self.roll_err_max * math.copysign(1, self.a_roll2)
        puls_roll: float = math.sqrt(abs(self.a_roll2) * self.aileron_limit / self.roll_err_max)  # rad.s^-1
        freq_roll: float = puls_roll / (2 * math.pi)  # Hz
        response_time_roll: float = 1 / freq_roll  # sec
        kd_roll: float = (2 * self.roll_damping * puls_roll - self.a_roll1) / self.a_roll2

        # ask if needed to plot the roots as a function of ki in something like simulink
        # (using small value as recommended in the book for now)
        ki_roll: float = 0.01

        # PI gains for the course angle hold control (outer loop)
        bw_factor_roll: int = 10  # bandwidth factor between roll and course
        puls_course: float = (1 / bw_factor_roll) * puls_roll  # rad.s^-1
        freq_course: float = puls_course / (2 * math.pi)  # Hz
        response_time_course: float = 1 / freq_course  # sec
        kp_course: float = 2 * self.course_damping * puls_course * (self.Va_trim / self.G)
        ki_course: float = puls_course ** 2 * (self.Va_trim / self.G)

        lat_pid_gains: dict[str, float] = {
            "kp_roll": kp_roll,
            "ki_roll": ki_roll,
            "kd_roll": kd_roll,
            "kp_course": kp_course,
            "ki_course": ki_course
        }

        lat_resp_times: dict[str, float] = {
            "roll": response_time_roll,
            "course": response_time_course
        }

        return lat_pid_gains, lat_resp_times


    def compute_long_pid_gains(self) -> tuple[dict[str, float], dict[str, float], float]:
        """
        Computes the PID gains for the longitudinal control system

        return: a tuple of two dictionaries: 
            - long_pid_gains: dict[str, float] : the first one containing the gains for the airspeed hold with commanded throttle +
        the second is for pitch angle (inner loop) and the third is for altitude hold with commanded pitch (outer loop)
            - long_resp_times: dict[str, float] : containing the response times of each loop
            - k_dc_pitch: float : DC gain of the pitch angle inner loop
        """
        # pitch attitude hold (inner loop)
        kp_pitch: float = self.elevator_limit / self.pitch_err_max * math.copysign(1, self.a_pitch3)
        puls_pitch: float = math.sqrt(self.a_pitch2 + (self.elevator_limit / self.pitch_err_max) * abs(self.a_pitch3))  # rad.s^-1
        freq_pitch: float = puls_pitch / (2 * math.pi)  # Hz
        response_time_pitch: float = 1 / freq_pitch  # sec
        kd_pitch: float = (2 * self.pitch_damping * puls_pitch - self.a_pitch1) / self.a_pitch3
        k_dc_pitch: float = (kp_pitch * self.a_pitch3) / (self.a_pitch2 + (kp_pitch * self.a_pitch3)) # DC Gain of the inner loop

        # Altitude hold using commanded pitch (outer loop)
        bw_factor_pitch: int = 10  # bandwidth factor between pitch inner loop and altitude hold outer loop
        puls_h: float = (1 / bw_factor_pitch) * puls_pitch  # rad.s^-1
        kp_h: float = (2 * self.h_damping * puls_h) / (k_dc_pitch * self.Va_trim)
        ki_h: float = puls_h ** 2 / (k_dc_pitch * self.Va_trim)
        freq_h: float = puls_h / (2 * math.pi)  # Hz
        response_time_h: float = 1 / freq_h  # sec

        # Airspeed hold using throttle
        kp_vth: float = (2 * self.v_damping * self.v_puls - self.av1) / self.av2
        ki_vth: float = (self.v_puls ** 2) / self.av2

        long_pid_gains: dict[str, float] = {
            "kp_vth": kp_vth, # airspeed commanded throttle : vth
            "ki_vth": ki_vth,
            "kp_pitch": kp_pitch, # pitch angle
            "kd_pitch": kd_pitch,
            "kp_h": kp_h, # altitude : h
            "ki_h": ki_h
        }

        long_resp_times: dict[str, float] = {
            "pitch": response_time_pitch,
            "alt": response_time_h
        }

        return long_pid_gains, long_resp_times, k_dc_pitch
