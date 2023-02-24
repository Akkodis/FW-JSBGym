import jsbsim

class AeroModel(object):
    def __init__(self, fdm: jsbsim.FGFDMExec):
        self.Ixx = fdm['inertia/ixx-slugs_ft2'] * 1.355818 # slugs*ft2 to kg*m2
        self.Iyy = fdm['inertia/iyy-slugs_ft2'] * 1.355818 
        self.Izz = fdm['inertia/izz-slugs_ft2'] * 1.355818
        self.Ixz = fdm['inertia/ixz-slugs_ft2'] * 1.355818
        self.Ixy = fdm['inertia/ixy-slugs_ft2'] * 1.355818
        self.Iyz = fdm['inertia/iyz-slugs_ft2'] * 1.355818
        self.air_density = fdm['atmosphere/rho-slugs_ft3'] * 515.379 # slugs/ft3 to kg/m3

        # gamma coefficients
        self.gamma = self.Ixx * self.Izz - self.Ixz**2
        self.gamma1 = (self.Ixz * (self.Ixx - self.Iyy + self.Izz)) / self.gamma
        self.gamma2 = (self.Izz * (self.Izz - self.Iyy) + self.Ixz**2) / self.gamma
        self.gamma3 = self.Izz / self.gamma
        self.gamma4 = self.Ixz / self.gamma
        self.gamma5 = (self.Izz - self.Ixx) / self.Iyy
        self.gamma6 = self.Ixz / self.Iyy
        self.gamma7 = ((self.Ixx - self.Iyy) * self.Ixx + self.Ixz**2) / self.gamma
        self.gamma8 = self.Ixx / self.gamma

        self.Va_trim = 65.0 / 3.6 # nominal Va km/h to m/s
        self.S = 0.75 # wing span m2
        self.b = 2.10 # wing span m

        self.Clp = -0.4042
        self.Cnp = 0.0044
        self.Clda = 0.1202
        self.Cnda = 0.1202

        self.Cpp = self.gamma3 * self.Clp + self.gamma4 * self.Cnp
        self.Cpda = self.gamma3 * self.Clda + self.gamma4 * self.Cnda

        # Roll TF coefficients
        self.a_roll1 = -1/2 * self.air_density * self.Va_trim**2 * self.S * self.b * self.Cpp * (self.b/(2 * self.Va_trim))
        self.a_roll2 = 1/2 * self.air_density * self.Va_trim**2 * self.S * self.b * self.Cpda
        print("a_roll1 = ", self.a_roll1)
        print("a_roll2 = ", self.a_roll2)
        
        self.aileron_limit = 30.0 # aileron actuator max deflection - deg
        self.roll_max = 20.0 # roll max angle - deg

    def update(self, fdm: jsbsim.FGFDMExec):
        self.Clp = fdm['aero/coefficient/Clp']
        self.Cnp = fdm['aero/coefficient/Cnp']
        self.Clda = fdm['aero/coefficient/Clda']
        self.Cnda = fdm['aero/coefficient/Cnda']

