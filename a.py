#!/usr/bin/env python3
import jsbsim
import os
import time
import csv

FG_OUT_FILE = 'flightgear.xml'
fdm = jsbsim.FGFDMExec(None)
# fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), FG_OUT_FILE))
fdm.load_ic('initial_conditions/basic_ic.xml', False)
fdm.load_model('x8')
# fdm.set_dt(1/60)
fdm.run_ic()

prop = fdm.query_property_catalog("position")
print(prop)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

fieldnames = ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']

# create flight_data csv file with header
with open('data/flight_data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

cnt = 0
while fdm.run():
    fdm.set_property_value("fcs/aileron-cmd-norm", -0.3)
    # fdm.set_property_value("fcs/elevator-cmd-norm", -0.1)
    fdm.set_property_value("fcs/throttle-cmd-norm", 0.1)

    latitude = fdm.get_property_value("position/lat-gc-deg")
    longitude = fdm.get_property_value("position/long-gc-deg")
    altitude = fdm.get_property_value("position/h-sl-meters")
    # altitude = fdm.get_property_value("position/h-agl-km")
    # print(f"lat: {latitude}, lon: {longitude}, alt: {altitude}")

    roll = fdm.get_property_value("attitude/roll-rad")
    pitch = fdm.get_property_value("attitude/pitch-rad")
    yaw = fdm.get_property_value("attitude/heading-true-rad")

    roll_rate = fdm.get_property_value("velocities/p-rad_sec")
    pitch_rate = fdm.get_property_value("velocities/q-rad_sec")
    yaw_rate = fdm.get_property_value("velocities/r-rad_sec")

    airspeed = fdm.get_property_value("velocities/vc-kts")

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

    time.sleep(0.01)
    # input(f"Press Enter to continue...{cnt}")
    cnt += 1
    pass
