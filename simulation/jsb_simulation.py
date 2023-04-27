#!/usr/bin/env python3
import jsbsim
import os
import subprocess
import time
from trim.trim_point import TrimPoint


class Simulation(object):
    FG_OUT_FILE = 'flightgear.xml'

    def __init__(self,
                 fdm_frequency: float = 120.0, # 120.0 Hz is the default frequency of the JSBSim FDM
                 aircraft_id: str = 'x8',
                 viz_time_factor: float = 1.0,
                 enable_fgear_viz: bool = False,
                 enable_trim: bool = False,
                 trim_point: TrimPoint = None
                 ) -> None:

        # initialization of some attributes
        self.fdm = jsbsim.FGFDMExec('fdm_descriptions') # provide the path of the fdm_descriptions folder containing the aircraft, engine, etc. .xml files
        self.fdm.set_debug_level(1)
        self.aircraft_id: str = aircraft_id
        self.fdm_dt: float = 1 / fdm_frequency
        self.viz_dt = None
        self.enable_trim: bool = enable_trim
        self.trim_point: TrimPoint = trim_point

        # set the FDM time step
        self.fdm.set_dt(self.fdm_dt)

        # load the aircraft model
        self.fdm.load_model(self.aircraft_id)

        # code for flightgear output here :
        if enable_fgear_viz:
            self.fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FG_OUT_FILE))
            self.fgear_viz = FlightGearVisualizer(self)

        # set the visualization time factor (plot and/or flightgear visualization)
        self.set_viz_time_factor(time_factor=viz_time_factor)

        # load and run initial conditions
        self.load_run_ic()


    def load_run_ic(self):
        # initialize the simulation:
        # if we start in trimmed flight, load those corresponding ic
        if self.enable_trim and self.trim_point is not None:
            self.fdm['ic/h-sl-ft'] = self.trim_point.h_ft # above sea level altitude
            self.fdm['ic/vc-kts'] = self.trim_point.Va_kts # ic airspeed
            self.fdm['ic/gamma-deg'] = self.trim_point.gamma_deg # steady level flight
        # if we start in untrimmed flight, load the basic ic
        else:
            ic_path = f'initial_conditions/{self.aircraft_id}_basic_ic.xml'
            self.fdm.load_ic(ic_path, False)

        success: bool = self.fdm.run_ic()
        if not success:
            raise RuntimeError("Failed to initialize the simulation.")
        return success

    def run_step(self) -> bool:
        result = self.fdm.run()
        if self.viz_dt is not None:
            time.sleep(self.viz_dt)
        return result


    def set_viz_time_factor(self, time_factor: float) -> None:
        if time_factor is None:
            self.viz_dt = None
        elif time_factor <= 0:
            raise ValueError("The time factor must be strictly positive.")
        else:
            self.viz_dt = self.fdm_dt / time_factor


class FlightGearVisualizer(object):
    TYPE = 'socket'
    DIRECTION = 'in'
    RATE = 60
    SERVER = ''
    PORT = 5550
    PROTOCOL = 'udp'
    LOADED_MESSAGE = "Primer reset to 0"
    TIME = 'noon'
    AIRCRAFT_FG_ID = 'c172p'

    def __init__(self, sim: Simulation) -> None:
        self.flightgear_process = self.launch_flightgear(aircraft_fgear_id=sim.aircraft_id)
        time.sleep(15)

    def launch_flightgear(self, aircraft_fgear_id: str = 'c172p') -> subprocess.Popen:
        # cmd for running flightgear(binary apt package version 2020.3.13) from terminal
        # cmd = f'fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft={aircraft_fgear_id} --timeofday=noon \
        # --disable-ai-traffic --disable-real-weather-fetch'

        # cmd for running flightgear(.AppImage version 2020.3.17) from terminal
        cmd: str = f'exec $HOME/Apps/FlightGear-2020.3.17/FlightGear-2020.3.17-x86_64.AppImage --fdm=null \
        --native-fdm=socket,in,60,,5550,udp --aircraft=c172p --timeofday=noon --disable-ai-traffic --disable-real-weather-fetch'

        flightgear_process = subprocess.Popen(cmd,
                                              shell=True,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT)
        print("Started FlightGear process with PID: ", flightgear_process.pid)
        while True:
            out = flightgear_process.stdout.readline().decode()
            if self.LOADED_MESSAGE in out:
                print("FlightGear loaded successfully.")
                break
            else:
                print(out.strip())
        return flightgear_process
