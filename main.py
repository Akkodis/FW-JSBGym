#!/usr/bin/env python3
import csv
from simulation import Simulation
import os

# create a simulation object
sim = Simulation(fdm_frequency_hz=60.0, aircraft_id='x8', viz_time_factor=1.0 ,enable_fgear_viz=False)

# properties = sim.fdm.query_property_catalog("position")
# sim.fdm.print_property_catalog()
# print("********PROPERTIES***********\n", properties)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# create flight_data csv file with header
with open('data/flight_data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'])
    csv_writer.writeheader()

# simulation loop
while sim.run_step():
    latitude = sim.fdm.get_property_value("position/lat-gc-deg")
    longitude = sim.fdm.get_property_value("position/long-gc-deg")
    altitude = sim.fdm.get_property_value("position/h-sl-meters")
    # print(f"lat: {latitude}, lon: {longitude}, alt: {altitude}")

    roll = sim.fdm.get_property_value("attitude/roll-rad")
    pitch = sim.fdm.get_property_value("attitude/pitch-rad")
    yaw = sim.fdm.get_property_value("attitude/heading-true-rad")
    # print(f"r: {roll}, p: {pitch}, y: {yaw}")

    # write flight data to csv
    with open('data/flight_data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'])
        info = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        }
        csv_writer.writerow(info)