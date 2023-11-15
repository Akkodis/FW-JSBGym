#!/usr/bin/env python3
import jsbsim
import os
import time
from jsbgym.trim.trim_point import TrimPoint
from typing import Union
from jsbgym.utils import jsbsim_properties as prp


class Simulation(object):
    """
        Simulation class. Wrapper class for the JSBSim FDM.
        Access to the FDM properties is done through the `__getitem__` and `__setitem__` methods, by only providing BoundedProperty or Property object as keys.
        Example: `sim[prp.airspeed_kts] = 30.0` sets the airspeed to 30 kts.

        Attr:
            - `fdm`: the JSBSim FDM object
            - `aircraft_id`: the aircraft to simulate
            - `fdm_dt`: the time step of the FDM
            - `viz_dt`: the time step of the visualization
            - `enable_trim`: whether to start the simulation in trimmed flight
            - `trim_point`: the trim point to start the simulation in
            - `FG_OUT_FILE`: the name of the file containing the FlightGear output protocol settings
    """
    FG_OUT_FILE = 'flightgear.xml'

    def __init__(self,
                 fdm_frequency: float,
                 aircraft_id: str = 'x8',
                 viz_time_factor: float = None,
                 enable_fgear_output: bool = False,
                 enable_trim: bool = False,
                 trim_point: TrimPoint = None
                 ) -> None:
        """
            Args:
                - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
                - `aircraft_id`: the aircraft to simulate
                - `viz_time_factor`: the factor by which the simulation time is scaled for visualization
                - `enable_fgear_output`: whether to enable FlightGear output for JSBSim <-> FGear communcation
                - `enable_trim`: whether to start the simulation in trimmed flight
                - `trim_point`: the trim point to start the simulation in
        """
        self.fdm = jsbsim.FGFDMExec('fdm_descriptions') # provide the path of the fdm_descriptions folder containing the aircraft, engine, etc. .xml files
        self.fdm.set_debug_level(0) # don't print debug info from JSBSim to avoid cluttering the output
        self.aircraft_id: str = aircraft_id
        self.fdm_dt: float = 1 / fdm_frequency
        self.viz_dt: float = None
        self.enable_trim: bool = enable_trim
        self.trim_point: TrimPoint = trim_point

        # set the FDM time step
        self.fdm.set_dt(self.fdm_dt)

        # load the aircraft model
        self.fdm.load_model(self.aircraft_id)

        # code for flightgear output here :
        if enable_fgear_output:
            self.fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FG_OUT_FILE))

        # set the visualization time factor (plot and/or flightgear visualization)
        self.set_viz_time_factor(time_factor=viz_time_factor)

        # load and run initial conditions
        self.load_run_ic()


    def __getitem__(self, prop: Union[prp.Property, prp.BoundedProperty] | str) -> float:
        if isinstance(prop, str):
            return self.fdm[prop]
        else:
            return self.fdm[prop.name]


    def __setitem__(self, prop: Union[prp.Property, prp.BoundedProperty] | str, value) -> None:
        if isinstance(prop, str):
            self.fdm[prop] = value
        else:
            self.fdm[prop.name] = value


    def load_run_ic(self) -> bool:
        """
            Load and run the initial conditions of the simulation.

            Returns:
                - `success`: whether the simulation was initialized successfully
        """
        # if we start in trimmed flight, load those corresponding ic
        if self.enable_trim and self.trim_point is not None:
            self.fdm['ic/h-sl-ft'] = self.trim_point.h_ft # above sea level altitude
            self.fdm['ic/vc-kts'] = self.trim_point.Va_kts # ic airspeed
            self.fdm['ic/gamma-deg'] = self.trim_point.gamma_deg # steady level flight
        # if we start in untrimmed flight, load the basic ic from file
        else:
            ic_path: str = f'initial_conditions/{self.aircraft_id}_basic_ic.xml'
            self.fdm.load_ic(ic_path, False)

        # error handling for ic loading
        success: bool = self.fdm.run_ic()
        if not success:
            raise RuntimeError("Failed to initialize the simulation.")
        return success


    def run_step(self) -> bool:
        """
            Run one step of the JSBSim simulation.
        """
        # run the simulation for one step
        result: bool = self.fdm.run()
        # sleep for the visualization time step if needed
        if self.viz_dt is not None:
            time.sleep(self.viz_dt)
        return result


    def set_viz_time_factor(self, time_factor: float) -> None:
        """
            Set the time factor for visualization.
        """
        if time_factor is None:
            self.viz_dt = None
        elif time_factor <= 0:
            raise ValueError("The time factor must be strictly positive.")
        else:
            # Convert the time factor into a time step period for visualization
            self.viz_dt = self.fdm_dt / time_factor

