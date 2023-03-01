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

        # Aero coeffs of the x8
        self.Clp = -0.4042
        self.Cnp = 0.0044
        self.Clda = 0.1202
        self.Cnda = 0.1202

        # intermediate values of aero coeffs weighted by inertia
        self.Cpp = self.gamma3 * self.Clp + self.gamma4 * self.Cnp
        self.Cpda = self.gamma3 * self.Clda + self.gamma4 * self.Cnda

        # Roll TF coefficients
        self.a_roll1 = -1 / 2 * self.air_density * self.Va_trim ** 2 * self.S * self.b * self.Cpp * (
                    self.b / (2 * self.Va_trim))
        self.a_roll2 = 1 / 2 * self.air_density * self.Va_trim ** 2 * self.S * self.b * self.Cpda
        print("a_roll1 = ", self.a_roll1)
        print("a_roll2 = ", self.a_roll2)

        self.aileron_limit = 30.0 * (math.pi / 180)  # aileron actuator max deflection : deg to rad
        self.roll_max = 45.0 * (math.pi / 180)  # roll max angle : deg to rad
        self.roll_err_max = self.roll_max * 2  # max expected error, roll_max * 2 : rad
        self.roll_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink

        self.course_damping = 1.5  # ask if needed to plot the step responses for various damping ratios in something
        # like simulink

        self.compute_lat_pid_gains()

        # longitutinal pitch TF coefficients

    def compute_lat_pid_gains(self):
        # PID gains for roll attitude control (inner loop)
        kp_roll = self.aileron_limit / self.roll_err_max * math.copysign(1, self.a_roll2)
        puls_roll = math.sqrt(abs(self.a_roll2) * self.aileron_limit / self.roll_err_max)  # rad.s^-1
        freq_roll = puls_roll / (2 * math.pi)  # Hz
        response_time_roll = 1 / freq_roll  # sec
        kd_roll = (2 * self.roll_damping * puls_roll - self.a_roll1) / self.a_roll2

        # ask if needed to plot the roots as a function of ki in something like simulink
        # (using small value as recommended in the book for now)
        ki_roll = 0.01

        # PI gains for the course angle hold control (outer loop)
        bw_factor = 5  # bandwidth factor
        puls_course = (1 / bw_factor) * puls_roll  # rad.s^-1
        freq_course = puls_course / (2 * math.pi)  # Hz
        response_time_course = 1 / freq_course  # sec
        kp_course = 2 * self.course_damping * puls_course * (self.Va_trim / self.G)
        kd_course = puls_course ** 2 * (self.Va_trim / self.G)
        print("end")

    def compute_long_pid_gains(self):
        # pitch attitude hold
        pass
