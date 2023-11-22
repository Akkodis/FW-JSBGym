import subprocess
import time
from jsbgym.simulation.jsb_simulation import Simulation


class PlotVisualizer(object):
    def __init__(self, scale: bool, telemetry_file: str) -> None:
        cmd: str = ""
        if scale:
            cmd = f"python visualizers/attitude_control_telemetry.py --tele-file {telemetry_file} --scale"
        else:
            cmd = f"python visualizers/attitude_control_telemetry.py --tele-file {telemetry_file}"
        print(telemetry_file)
        self.process: subprocess.Popen = subprocess.Popen(cmd, 
                                                          shell=True,
                                                          stdout=subprocess.PIPE,
                                                          stderr=subprocess.STDOUT)
        print("Started attitude_control_telemetry.py process with PID: ", self.process.pid)
        while True:
            out: str = self.process.stdout.readline().decode()
            print(out.strip())
            if "Animation plot started..." in out:
                print("attitude_control_telemetry.py loaded successfully.")
                break


class FlightGearVisualizer(object):
    TYPE = 'socket'
    DIRECTION = 'in'
    RATE = 60
    SERVER = ''
    PORT = 5550
    PROTOCOL = 'udp'
    LOADED_MESSAGE = "PNG lib warning : Malformed iTXt chunk"
    TIME = 'noon'
    AIRCRAFT_FG_ID = 'c172p'

    def __init__(self, sim: Simulation) -> None:
        # launching flightgear with the corresponding aircraft_id
        self.flightgear_process = self.launch_flightgear(aircraft_fgear_id=sim.aircraft_id)
        time.sleep(15)

    def launch_flightgear(self, aircraft_fgear_id: str = 'c172p') -> subprocess.Popen:
        # cmd for running flightgear(binary apt package version 2020.3.13) from terminal
        # cmd = f'fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft=c172p --timeofday=noon \
        # --disable-ai-traffic --disable-real-weather-fetch'

        # cmd for running flightgear(.AppImage version 2020.3.17) from terminal.
        # We ignore the aircraft_id to load the c172p viz, since x8 doesn't exist in fgear
        cmd: str = f'exec $HOME/Apps/FlightGear-2020.3.17/FlightGear-2020.3.17-x86_64.AppImage --fdm=null \
        --native-fdm=socket,in,60,,5550,udp --aircraft={aircraft_fgear_id} --timeofday=noon --disable-ai-traffic --disable-real-weather-fetch'

        flightgear_process = subprocess.Popen(cmd,
                                              shell=True,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT)
        print("Started FlightGear process with PID: ", flightgear_process.pid)
        while True:
            out: str = flightgear_process.stdout.readline().decode()
            if self.LOADED_MESSAGE in out:
                print("FlightGear loaded successfully.")
                break
            else:
                print(out.strip())
        return flightgear_process
