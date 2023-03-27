import yaml
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import loader

class TrimPoint(object):

    def __init__(self, aircraft_id: str = "x8"):
        stream = open(f"trim/trim_points_{aircraft_id}.yaml", "r")
        dictionary = yaml.load_all(stream, Loader=Loader)
        key_count = 0
        for doc in dictionary:
            # print("New document:")
            for key, value in doc.items():
                # print(key + " : " + str(value))
                # if type(value) is list:
                #     print(str(len(value)))
                match key:
                    case 'Va':
                        self.Va: float = value
                        key_count += 1
                    case 'AoA':
                        self.alpha: float = value
                        key_count += 1
                    case 'gamma':
                        self.gamma: float = value
                        key_count += 1
                    case 'h':
                        self.h: float = value
                        key_count += 1
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