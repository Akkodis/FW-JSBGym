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
                 allow_fgear_output: bool = True) -> None:
        
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(1)

        # code for flightgear output here :
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FG_OUT_FILE))
        self.fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FG_OUT_FILE))

        self.fdm_dt = 1 / fdm_frequency_hz
        ic_path = 'initial_conditions/basic_ic.xml'
        self.fdm.load_ic(ic_path, False)
        self.fdm.load_model(aircraft_id)
        self.fdm.set_dt(self.fdm_dt)
        success = self.fdm.run_ic()
        if not success:
            raise RuntimeError("Failed to initialize the simulation.")
        # self.fdm.enable_output()
        self.sim_dt = None

    def run_step(self) -> bool:
        result = self.fdm.run()
        # if self.sim_dt is not None:
        #     print("sim_dt: ", self.sim_dt)
        #     time.sleep(self.sim_dt)
        return result

    def set_sim_time_factor(self, time_factor: float) -> None:
        if time_factor is None:
            self.sim_dt = None
        elif time_factor <= 0:
            raise ValueError("The time factor must be strictly positive.")
        else:
            self.sim_dt = self.fdm_dt / time_factor

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

    def __init__(self, sim: Simulation, fg_time_factor: float = 1.0) -> None:
        self.fg_time_factor = fg_time_factor
        sim.set_sim_time_factor(self.fg_time_factor)
        self.flightgear_process = self.launch_flightgear()
        time.sleep(40)

    def launch_flightgear(self, aircraft_fg_id: str = 'c172p'):
        cmd = 'fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft=c172p --timeofday=noon --disable-ai-traffic --disable-real-weather-fetch'

        flightgear_process = subprocess.Popen(cmd,
                                              shell=True,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT)
        print("Started FlightGear process with PID: ", flightgear_process.pid)
        while True:
            out = flightgear_process.stdout.readline().decode()
            # err = flightgear_process.stderr.readline().decode()
            if self.LOADED_MESSAGE in out:
                print("FlightGear loaded successfully.")
                break
            else:
                print(out.strip())
                # print(err.strip())
            rc = flightgear_process.poll()
        return flightgear_process

