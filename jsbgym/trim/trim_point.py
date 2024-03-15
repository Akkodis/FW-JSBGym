import yaml
from numpy import rad2deg
from pkg_resources import resource_filename


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import loader

class TrimPoint(object):

    def __init__(self, aircraft_id: str = "x8"):
        trim_cfg_path = resource_filename("jsbgym", f"trim/trim_points_{aircraft_id}.yaml")
        stream = open(trim_cfg_path, "r")
        dictionary = yaml.load_all(stream, Loader=Loader)
        key_count = 0
        for doc in dictionary:
            for key, value in doc.items():
                match key:
                    case 'Va': # trim airspeed
                        self.Va_kts: float = value # kts
                        self.Va_mps: float = self.Va_kts / 1.944 # mps
                        self.Va_kph: float = self.Va_kts * 1.852 # kph
                        key_count += 1
                    case 'AoA': # angle of attack
                        self.alpha_deg: float = value # deg
                        self.alpha_rad: float = rad2deg(self.alpha_deg) # rad
                        key_count += 1
                    case 'gamma': # flight path angle
                        self.gamma_deg: float = value # deg
                        self.gamma_rad: float = rad2deg(self.gamma_deg) # rad
                        key_count += 1
                    case 'h': # z position / above sea level altitude
                        self.h_ft: float = value # ft
                        self.h_m: float = self.h_ft / 3.281 # m
                        self.h_km: float = self.h_ft / 3281 # km
                        key_count += 1
                    # input commands on fcs
                    case 'throttle':
                        self.throttle: float = value
                        key_count += 1
                    case 'elevator':
                        self.elevator: float = value
                        key_count += 1
                    case 'aileron':
                        self.aileron: float = value
                        key_count += 1
                    case 'rudder':
                        self.rudder: float = value
                        key_count += 1
                    case _:
                        print("[Warning] Unknown key: " + key)

        if key_count != 8:
            print("[Warning] Trim point yaml file is missing a key !")

        # no wind:
        self.theta_deg: float = self.alpha_deg + self.gamma_deg
        self.theta_rad: float = self.alpha_rad + self.gamma_rad
