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
        print("AAAAAAAAAAAAAAAAAAAAAAA CONNARD")
