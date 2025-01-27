#!/usr/bin/env python3
import resource
import jsbsim
import os
import time
import numpy as np
from fw_jsbgym.trim.trim_point import TrimPoint
from typing import Union, Tuple
from fw_jsbgym.utils import jsbsim_properties as prp
from enum import Enum
from pkg_resources import resource_filename
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty


class ConvFactor(Enum):
    kts2mps = 0.51444
    kts2kph = 1.852
    fps2mps = 0.3048
    fps2kph = 1.09728
    kph2mps = 0.277778
    ft2m = 0.3048


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
        fdm_descriptions_path = resource_filename("fw_jsbgym", "fdm_descriptions")
        self.fdm = jsbsim.FGFDMExec(fdm_descriptions_path) # provide the path of the fdm_descriptions folder containing the aircraft, engine, etc. .xml files
        self.fdm.set_debug_level(0) # don't print debug info from JSBSim to avoid cluttering the output
        self.aircraft_id: str = aircraft_id
        self.fdm_dt: float = 1 / fdm_frequency
        self.viz_dt: float = None
        self.enable_trim: bool = enable_trim
        self.trim_point: TrimPoint = trim_point

        self.prp_conv_fps2mps: Tuple[BoundedProperty, ...] = (
            prp.windspeed_north_mps, prp.windspeed_east_mps, prp.windspeed_down_mps,
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps,
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps
        )

        self.prp_conv_fps2kph: Tuple[BoundedProperty, ...] = (
            prp.windspeed_north_kph, prp.windspeed_east_kph, prp.windspeed_down_kph,
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph,
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph
        )

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


    def __getitem__(self, prop: Union[prp.Property, prp.HelperProperty, prp.BoundedProperty, prp.BoundedHelperProperty] | str) -> float:
        self.convert_props_to_SI(prop)
        if isinstance(prop, str):
            return self.fdm[prop]
        else:
            return self.fdm[prop.name]


    def __setitem__(self, prop: Union[prp.Property, prp.HelperProperty, prp.BoundedProperty, prp.BoundedHelperProperty] | str, value) -> None:
        if isinstance(prop, str):
            self.fdm[prop] = value
        else:
            self.fdm[prop.name] = value
        self.convert_props_to_SI(prop)


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
            ic_path: str = resource_filename('fw_jsbgym', f'initial_conditions/{self.aircraft_id}_basic_ic.xml')
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


    def convert_props_to_SI(self, prop) -> None:
        """
            Converts some properties from imperial to international metric system
        """
        self.fdm[prp.zero.name] = 0.0
        self.fdm[prp.zero_.name] = 0.0
        if not isinstance(prop, str):
            if prop in self.prp_conv_fps2mps:
                self.fdm[prop.name] = self.fdm[prop.name[:-3]+'fps'] * ConvFactor.fps2mps.value
            elif prop in self.prp_conv_fps2kph:
                self.fdm[prop.name] = self.fdm[prop.name[:-3]+'fps'] * ConvFactor.fps2kph.value
            elif prop == prp.airspeed_mps or prop == prp.airspeed_kph:
                self.fdm[prop.name] = self.fdm[prp.airspeed_kts.name] * ConvFactor.kts2mps.value
                self.fdm[prop.name] = self.fdm[prp.airspeed_kts.name] * ConvFactor.kts2kph.value
            elif 'enu' in prop.name:
                lat_gc = self.fdm[prp.lat_gc_deg.name]
                lon_gc = self.fdm[prp.lng_gc_deg.name]
                alt = self.fdm[prp.altitude_sl_m.name]
                enu_coords = self.geocentric2enu(lat_gc, lon_gc, alt, 47.635784, 2.460938, 0.0)
                self.fdm[prp.enu_x_m.name] = enu_coords[0]
                self.fdm[prp.enu_y_m.name] = enu_coords[1]
                self.fdm[prp.enu_z_m.name] = enu_coords[2]
                self.fdm[prp.enu_x_km.name] = enu_coords[0] / 1000
                self.fdm[prp.enu_y_km.name] = enu_coords[1] / 1000
                self.fdm[prp.enu_z_km.name] = enu_coords[2] / 1000


    def geocentric2ecef(self, lat, lon, alt):
        """
        Convert geocentric coordinates to ECEF coordinates
        args: lat, lon, alt (in degrees, degrees, meters)
        """
        # Convert degrees to radians
        lat = np.radians(lat)
        lon = np.radians(lon)

        # WGS-84 ellipsoid constant: semi-major axis (approx earth radius in m)
        r = 6378137.0

        # Compute ECEF coordinates
        x = (r + alt) * np.cos(lat) * np.cos(lon)
        y = (r + alt) * np.cos(lat) * np.sin(lon)
        z = (r + alt) * np.sin(lat)

        return np.array([x, y, z])


    def ecef2enu(self, x, y, z, lat0, lon0, alt0):
        """
        Convert ECEF coordinates to ENU coordinates
        args: x, y, z, lat0, lon0, alt0 (in meters, degrees, degrees, meters)
        """
        # Reference point (in ECEF)
        ref_ecef = self.geocentric2ecef(lat0, lon0, alt0)

        # Translation vector
        dx, dy, dz = x - ref_ecef[0], y - ref_ecef[1], z - ref_ecef[2]

        # Rotation matrix
        ref_lat = np.radians(lat0)
        ref_lon = np.radians(lon0)

        R = np.array([
            [-np.sin(ref_lon), np.cos(ref_lon), 0],
            [-np.sin(ref_lat) * np.cos(ref_lon), -np.sin(ref_lat) * np.sin(ref_lon), np.cos(ref_lat)],
            [np.cos(ref_lat) * np.cos(ref_lon), np.cos(ref_lat) * np.sin(ref_lon), np.sin(ref_lat)]
        ])

        enu = R @ np.array([dx, dy, dz])

        return enu
    

    def geocentric2enu(self, lat, lon, alt, lat0, lon0, alt0):
        """
        Convert geocentric coordinates to ENU coordinates
        args: lat[deg], lon[deg], alt[m], lat0[deg], lon0[deg], alt0[m]
        """
        ecef = self.geocentric2ecef(lat, lon, alt)
        enu = self.ecef2enu(ecef[0], ecef[1], ecef[2], lat0, lon0, alt0)
        return enu
