import jsbsim
import math


class AeroModel(object):
    G = 9.82  # m/s2

    def __init__(self, fdm: jsbsim.FGFDMExec):
        self.Ixx = fdm['inertia/ixx-slugs_ft2'] * 1.355818  # slugs*ft2 to kg*m2
        self.Iyy = fdm['inertia/iyy-slugs_ft2'] * 1.355818
        self.Izz = fdm['inertia/izz-slugs_ft2'] * 1.355818
        self.Ixz = fdm['inertia/ixz-slugs_ft2'] * 1.355818
        self.Ixy = fdm['inertia/ixy-slugs_ft2'] * 1.355818
        self.Iyz = fdm['inertia/iyz-slugs_ft2'] * 1.355818
        self.air_density = fdm['atmosphere/rho-slugs_ft3'] * 515.379  # slugs/ft3 to kg/m3
        self.mass = fdm['inertia/mass-slugs'] * 14.594 # slugs to kg

        # gamma coefficients
        self.gamma = self.Ixx * self.Izz - self.Ixz ** 2
        self.gamma1 = (self.Ixz * (self.Ixx - self.Iyy + self.Izz)) / self.gamma
        self.gamma2 = (self.Izz * (self.Izz - self.Iyy) + self.Ixz ** 2) / self.gamma
        self.gamma3 = self.Izz / self.gamma
        self.gamma4 = self.Ixz / self.gamma
        self.gamma5 = (self.Izz - self.Ixx) / self.Iyy
        self.gamma6 = self.Ixz / self.Iyy
        self.gamma7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz ** 2) / self.gamma
        self.gamma8 = self.Ixx / self.gamma

        self.Va_trim = 65.0 / 3.6  # nominal Va km/h to m/s
        self.S = 0.75  # wing span m2
        self.b = 2.10  # wing span m
        self.c = 0.3571 # mean aerodynamic chord m

        # Aero coeffs of the x8
        self.Clp = -0.4042
        self.Cnp = 0.0044
        self.Clda = 0.1202
        self.Cnda = 0.1202
        self.Cmq = -1.3012
        self.Cma = -0.4629
        self.Cmde = -0.2292
        self.CDo = 0.0197
        self.CDa = 0.0791
        self.CDde = 0.0633

        # intermediate values of aero coeffs weighted by inertia
        self.Cpp = self.gamma3 * self.Clp + self.gamma4 * self.Cnp
        self.Cpda = self.gamma3 * self.Clda + self.gamma4 * self.Cnda

        # Roll TF coefficients
        self.a_roll1 = -1 / 2 * self.air_density * self.Va_trim ** 2 * self.S * self.b * self.Cpp * (
                        self.b / (2 * self.Va_trim))
        self.a_roll2 = 1 / 2 * self.air_density * self.Va_trim ** 2 * self.S * self.b * self.Cpda

        self.aileron_limit = 30.0 * (math.pi / 180)  # aileron actuator max deflection : deg to rad
        self.roll_max = 45.0 * (math.pi / 180)  # roll max angle : deg to rad
        self.roll_err_max = self.roll_max * 2  # max expected error, roll_max * 2 : rad
        self.roll_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink
        self.course_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink


        # longitutinal TF coefficients
        # pitch
        self.a_pitch1 = - ((self.air_density * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cmq * (self.c / (2 * self.Va_trim))
        self.a_pitch2 = - ((self.air_density * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cma
        self.a_pitch3 = ((self.air_density * self.Va_trim**2 * self.c * self.S) / (2 * self.Iyy)) * self.Cmde

        # airspeed
        # self.a_v1 = ((self.air_density * self.Va_trim * self.S) / self.mass) * (self.CDo + self.CDa * )

        self.elevator_limit = 30.0 * (math.pi / 180)  # elevator actuator max deflection : deg to rad
        self.pitch_max = 45.0 * (math.pi / 180)  # pitch max angle : deg to rad
        self.pitch_err_max = self.pitch_max * 2  # max expected error, pitch_max * 2 : rad
        self.pitch_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        self.h_damping = 1.5 # same as above but for altitude
        self.compute_long_pid_gains()

    def compute_lat_pid_gains(self) -> tuple[dict[str, float], dict[str, float]]:
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
        bw_factor_roll: int = 5  # bandwidth factor between roll and course
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
            "kd_course": ki_course
        }

        lat_resp_times: dict[str, float] = {
            "roll": response_time_roll,
            "course": response_time_course
        }

        return lat_pid_gains, lat_resp_times


    def compute_long_pid_gains(self):
        # pitch attitude hold (inner loop)
        kp_pitch = self.elevator_limit / self.pitch_err_max * math.copysign(1, self.a_pitch3)
        puls_pitch = math.sqrt(self.a_pitch2 + (self.elevator_limit / self.pitch_err_max) * abs(self.a_pitch3))  # rad.s^-1
        freq_pitch = puls_pitch / (2 * math.pi)  # Hz
        response_time_pitch = 1 / freq_pitch  # sec
        kd_pitch = (2 * self.pitch_damping * puls_pitch - self.a_pitch1) / self.a_pitch3
        k_dc_pitch = (kp_pitch * self.a_pitch3) / (self.a_pitch2 + (kp_pitch * self.a_pitch3)) # DC Gain of the inner loop

        # Altitude hold using commanded pitch (outer loop)
        bw_factor_pitch = 5  # bandwidth factor between pitch inner loop and altitude hold outer loop
        puls_h = (1 / bw_factor_pitch) * puls_pitch  # rad.s^-1
        kp_h = (2 * self.h_damping * puls_h) / (k_dc_pitch * self.Va_trim)
        ki_h = puls_h ** 2 / (k_dc_pitch * self.Va_trim)
        freq_h = puls_h / (2 * math.pi)  # Hz
        response_time_h = 1 / freq_h  # sec
        pass
