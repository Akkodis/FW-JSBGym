#!/usr/bin/env python3
import jsbsim
import os
import subprocess
import time


class Simulation(object):
    FG_OUT_FILE = 'flightgear.xml'

    def __init__(self,
                 fdm_frequency_hz: float = 60.0,
                 aircraft_id: str = 'c172p',
                 viz_time_factor: float = 1.0,
                 enable_fgear_viz: bool = False) -> None:

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(1)
        self.aircraft_id = aircraft_id
        self.fdm_dt = 1 / fdm_frequency_hz
        self.viz_dt = None

        # code for flightgear output here :
        if enable_fgear_viz:
            self.fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FG_OUT_FILE))
            self.fgear_viz = FlightGearVisualizer(self)

        # set the visualization time factor (plot and/or flightgear visualization)
        self.set_viz_time_factor(time_factor=viz_time_factor)

        # initialize the simulation : load aircraft model, load initial conditions
        ic_path = f'initial_conditions/{aircraft_id}_basic_ic.xml'
        self.fdm.load_ic(ic_path, False)
        self.fdm.load_model(aircraft_id)
        self.fdm.set_dt(self.fdm_dt)
        success = self.fdm.run_ic()
        if not success:
            raise RuntimeError("Failed to initialize the simulation.")

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
        time.sleep(2)

    def launch_flightgear(self, aircraft_fgear_id: str = 'c172p') -> subprocess.Popen:
        ## cmd for running flightgear(binary apt package version 2020.3.13) from terminal
        # cmd = f'fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft={aircraft_fgear_id} --timeofday=noon \
        # --disable-ai-traffic --disable-real-weather-fetch'

        ## cmd for running flightgear(.AppImage version 2020.3.17) from terminal
        cmd = f'exec $HOME/Apps/FlightGear-2020.3.17/FlightGear-2020.3.17-x86_64.AppImage --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft={aircraft_fgear_id} --timeofday=noon \
        --disable-ai-traffic --disable-real-weather-fetch'

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
