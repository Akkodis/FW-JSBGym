# TrimEnvelope.py
#
# Calculate the set of trim points for an aircraft over a range of airspeeds
# and range of flight path angles (gamma). The required thrust and AoA is
# indicated via a colormap for each trim point.
#
# Copyright (c) 2023 Sean McLeod
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>
#

import jsbsim
import matplotlib.pyplot as plt
import math
import csv
from time import sleep

# Global variables that must be modified to match your particular need
# The aircraft name
# Note - It should match the exact spelling of the model file
AIRCRAFT_NAME="x8"

# Path to JSBSim files, location of the folders "aircraft", "engines" and "systems"

# Avoid flooding the console with log messages
# jsbsim.FGJSBBase().debug_lvl = 0

fdm = jsbsim.FGFDMExec(None)

# Load the aircraft model
fdm.load_model(AIRCRAFT_NAME)

fdm.print_property_catalog()

# Set engines running
# fdm['propulsion/set-running'] = -1
fdm['propulsion/engine/set-running'] = 1

# Set alpha range for trim solutions
fdm['aero/alpha-max-rad'] = math.radians(12)
fdm['aero/alpha-min-rad'] = math.radians(-4.0)

# Trim results
results = []

# Find trim point for these particular initial conditions
fdm['ic/h-sl-ft'] = 1960
fdm['ic/vc-kts'] = 33 # ic speed 60 kmh
fdm['ic/gamma-deg'] = 0 # steady level flight

# Initialize the aircraft with initial conditions
fdm.run_ic()

# Trim
try:
    fdm['simulation/do_simple_trim'] = 1
    results.append((fdm['velocities/vc-kts'], fdm['aero/alpha-deg'], 0, fdm['fcs/throttle-cmd-norm'], fdm['fcs/pitch-trim-cmd-norm'], fdm['fcs/aileron-cmd-norm'], fdm['fcs/rudder-cmd-norm']))
except RuntimeError as e:
    # The trim cannot succeed. Just make sure that the raised exception
    # is due to the trim failure otherwise rethrow.
    if e.args[0] != 'Trim Failed':
        raise

# print("results throttle = ", results[0][3])
airspeed, alpha, gamma, throttle, elevator, aileron, rudder = zip(*results)

print(f"Airspeed = {airspeed[0]}    AoA = {alpha[0]}   Throttle = {throttle[0]}    Elevator = {elevator[0]}    Aileron = {aileron[0]}    Rudder = {rudder[0]}")

fieldnames: list[str] = ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']

# create flight_data csv file with header
with open('../data/flight_data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# run the simulation
timestep: int = 0
while fdm.run() and timestep < 200000:

    # reading values for plotting
    latitude: float = fdm["position/lat-gc-deg"]
    longitude: float = fdm["position/long-gc-deg"]
    altitude: float = fdm["position/h-sl-meters"]

    roll: float = fdm["attitude/roll-rad"]
    pitch: float = fdm["attitude/pitch-rad"]
    heading: float = fdm["attitude/heading-true-rad"]
    psi: float = fdm["attitude/psi-rad"]
    psi_gt: float = fdm["flight-path/psi-gt-rad"]

    roll_rate: float = fdm["velocities/p-rad_sec"]
    pitch_rate: float = fdm["velocities/q-rad_sec"]
    yaw_rate: float = fdm["velocities/r-rad_sec"]

    airspeed: float = fdm["velocities/vc-kts"]*1.852 # to m/s

    # write flight data to csv (for plotting)
    with open('../data/flight_data.csv', 'a') as csv_file:
        csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info: dict[str, float] = {
            fieldnames[0]: latitude,
            fieldnames[1]: longitude,
            fieldnames[2]: altitude,
            fieldnames[3]: roll,
            # "heading-true-rad": heading,
            # fieldnames[4]: course_angle,
            fieldnames[4]: pitch,
            # "psi-rad": psi,
            fieldnames[5]: heading,
            # "psi-gt-rad": psi_gt,
            fieldnames[6]: roll_rate,
            fieldnames[7]: pitch_rate,
            fieldnames[8]: yaw_rate,
            fieldnames[9]: airspeed
        }
        csv_writer.writerow(info)

    timestep += 1
    # sleep(0.01)