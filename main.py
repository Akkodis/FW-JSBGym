#!/usr/bin/env python3
import time
import csv
from simulation import Simulation, FlightGearVisualizer

# create a simulation object
sim = Simulation(fdm_frequency_hz=60.0, aircraft_id="c172p")
# properties = sim.fdm.query_property_catalog("position")
# sim.fdm.print_property_catalog()
# print("********PROPERTIES***********\n", properties)

fg_viz = FlightGearVisualizer(sim, fg_time_factor=1.6)

with open('data/flight_data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'])
    csv_writer.writeheader()

# for _ in range(9000):
run = True
while sim.run_step():
    latitude = sim.fdm.get_property_value("position/lat-gc-deg")
    longitude = sim.fdm.get_property_value("position/long-gc-deg")
    altitude = sim.fdm.get_property_value("position/h-sl-meters")
    # print(f"lat: {latitude}, lon: {longitude}, alt: {altitude}")

    roll = sim.fdm.get_property_value("attitude/roll-rad")
    pitch = sim.fdm.get_property_value("attitude/pitch-rad")
    yaw = sim.fdm.get_property_value("attitude/heading-true-rad")
    time.sleep(0.1)
    # print(f"r: {roll}, p: {pitch}, y: {yaw}")

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