#!/usr/bin/env python3
import csv
from simulation import Simulation
import os

# create a simulation object
sim = Simulation(fdm_frequency_hz=240.0, # going up to 240Hz solves some spin instability issues -> NaNs
                 aircraft_id='x8',
                 viz_time_factor=2.0,
                 enable_fgear_viz=False)

properties = sim.fdm.query_property_catalog("atmosphere")
# sim.fdm.print_property_catalog()
print("********PROPERTIES***********\n", properties)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

fieldnames = ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']

# create flight_data csv file with header
with open('data/flight_data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# simulation loop
while sim.run_step():

    # sim.fdm.set_property_value("fcs/aileron-cmd-norm", -0.3)
    # sim.fdm.set_property_value("fcs/elevator-cmd-norm", -0.1)
    # sim.fdm.set_property_value("fcs/throttle-cmd-norm", 0.3)

    latitude = sim.fdm.get_property_value("position/lat-gc-deg")
    longitude = sim.fdm.get_property_value("position/long-gc-deg")
    altitude = sim.fdm.get_property_value("position/h-sl-meters")

    roll = sim.fdm.get_property_value("attitude/roll-rad")
    pitch = sim.fdm.get_property_value("attitude/pitch-rad")
    yaw = sim.fdm.get_property_value("attitude/heading-true-rad")

    roll_rate = sim.fdm.get_property_value("velocities/p-rad_sec")
    pitch_rate = sim.fdm.get_property_value("velocities/q-rad_sec")
    yaw_rate = sim.fdm.get_property_value("velocities/r-rad_sec")

    airspeed = sim.fdm.get_property_value("velocities/vc-kts")

    # write flight data to csv
    with open('data/flight_data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "airspeed": airspeed
        }
        csv_writer.writerow(info)
