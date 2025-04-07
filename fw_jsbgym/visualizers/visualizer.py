import subprocess
import time
import matplotlib.pyplot as plt
from fw_jsbgym.simulation.jsb_simulation import Simulation
from pkg_resources import resource_filename


class PlotVisualizer(object):
    def __init__(self, animate: bool, env_id: str,  telemetry_file: str) -> None:
        viz_plot_path: str = ""
        self.env_id = env_id
        self.telemetry_file = telemetry_file

        if "AC" in env_id:
            viz_plot_path = resource_filename('fw_jsbgym', 'visualizers/attitude_control_telemetry.py')
        elif "Waypoint" in env_id or "Path" in env_id or \
            "CourseAltTracking" in env_id:
            viz_plot_path = resource_filename('fw_jsbgym', 'visualizers/waypoint_tracking_telemetry.py')
        assert viz_plot_path != "", "Invalid env_id. Please provide a valid env_id."

        cmd: str = ""
        print("Telemetry file: ", telemetry_file)
        # if animate is True, we run the animation plot in a separate process
        if animate:
            cmd = f"python {viz_plot_path} --tele-file {telemetry_file} --animate"

            self.process: subprocess.Popen = subprocess.Popen(cmd, 
                                                            shell=True,
                                                            stdout=subprocess.PIPE,
                                                            stderr=subprocess.STDOUT)
            print(f"Started {viz_plot_path} process with PID: {self.process.pid}")
            while True:
                out: str = self.process.stdout.readline().decode()
                print(out.strip())
                if "Animation plot started..." in out:
                    print("Animation plot loaded successfully.")
                    break

    def plot(self) -> None:
        if "AC" in self.env_id:
            # we run the attitude control telemetry plot in the current process
            from fw_jsbgym.visualizers.attitude_control_telemetry import setup_axes, animate
            ax = setup_axes()
            animate(0, ax, self.telemetry_file)
        elif "Waypoint" in self.env_id or "Path" in self.env_id or \
            "CourseAltTracking" in self.env_id:
            # we run the waypoint tracking telemetry plot in the current process
            from fw_jsbgym.visualizers.waypoint_tracking_telemetry import setup_axes, animate
            ax = setup_axes()
            animate(0, ax, self.telemetry_file)
            plt.show()

class FlightGearVisualizer(object):
    TYPE = 'socket'
    DIRECTION = 'in'
    RATE = 60
    SERVER = ''
    PORT = 5550
    PROTOCOL = 'udp'
    LOADED_MESSAGE = "setWeight() - not supported for null"
    TIME = 'noon'
    START_LAT = 47.635784
    START_LON = 2.460938
    START_ALT = 600

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
        --native-fdm={self.TYPE},{self.DIRECTION},{self.RATE},{self.SERVER},{self.PORT},{self.PROTOCOL} \
        --aircraft={aircraft_fgear_id} --timeofday={self.TIME} --lat={self.START_LAT} --lon={self.START_LON} --altitude={self.START_ALT} \
        --disable-ai-traffic --disable-real-weather-fetch --enable-terrasync'

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
