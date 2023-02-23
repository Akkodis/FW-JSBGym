#!/usr/bin/env python3
import csv
from simulation import Simulation
import os
import argparse
import random

# parse command line arguments
parser = argparse.ArgumentParser(description='Run JSBSim simulation.')
parser.add_argument('--fgear_viz', action='store_true', help='Enable FlightGear visualization.')
parser.add_argument('--plot_viz', action='store_true', help='Enable plot visualization.')
parser.add_argument('--aircraft_id', type=str, default='x8', help='Aircraft ID.')
parser.add_argument('--fdm_frequency', type=float, default=240.0, help='FDM frequency in Hz.')
parser.add_argument('--viz_time_factor', type=float, default=2.0, help='Visualization time factor.')
parser.add_argument('--flight_data', type=str, default='data/flight_data.csv', help='Path to flight data csv file.')
parser.add_argument('--turb', action='store_true', help='Enable turbulence.')
parser.add_argument('--wind', action='store_true', help='Enable wind.')
parser.add_argument('--gust', action='store_true', help='Enable gust.')
args = parser.parse_args()


# create a simulation object
sim = Simulation(fdm_frequency=args.fdm_frequency, # going up to 240Hz solves some spin instability issues -> NaNs
                 aircraft_id=args.aircraft_id,
                 viz_time_factor=args.viz_time_factor,
                 enable_fgear_viz=args.fgear_viz)

properties = sim.fdm.query_property_catalog("atmosphere")
# sim.fdm.print_property_catalog()
# print("********PROPERTIES***********\n", properties)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

fieldnames = ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']

# create flight_data csv file with header
with open(args.flight_data, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# set seed for random number generator
rand_seed = random.randint(0, 1000000)
sim.fdm.set_property_value("simulation/randomseed", rand_seed)
print("rho = ", sim.fdm.get_property_value("atmosphere/rho-slugs_ft3"))

# set wind
if args.wind:
    # sim.fdm.set_property_value("atmosphere/wind-north-fps", 100)
    sim.fdm.set_property_value("atmosphere/wind-east-fps", 40)

# set turbulences
if args.turb:
    sim.fdm.set_property_value("atmosphere/turb-type", 4) # Tustin turbulence type
    sim.fdm.set_property_value("atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps", 30)
    sim.fdm.set_property_value("atmosphere/turbulence/milspec/severity", 3)

# set wind gust
if args.gust:
    sim.fdm.set_property_value("atmosphere/cosine-gust/startup-duration-sec", 2)
    sim.fdm.set_property_value("atmosphere/cosine-gust/steady-duration-sec", 2)
    sim.fdm.set_property_value("atmosphere/cosine-gust/end-duration-sec", 2)
    sim.fdm.set_property_value("atmosphere/cosine-gust/magnitude-ft_sec", 40)
    sim.fdm.set_property_value("atmosphere/cosine-gust/frame", 2)
    sim.fdm.set_property_value("atmosphere/cosine-gust/X-velocity-ft_sec", 1)
    sim.fdm.set_property_value("atmosphere/cosine-gust/Y-velocity-ft_sec", 0)
    sim.fdm.set_property_value("atmosphere/cosine-gust/Z-velocity-ft_sec", 0)

timestep = 0

# simulation loop
while sim.run_step() and timestep < 20000:

    # sim.fdm.set_property_value("fcs/aileron-cmd-norm", -0.3)
    # sim.fdm.set_property_value("fcs/elevator-cmd-norm", -0.05)
    sim.fdm.set_property_value("fcs/throttle-cmd-norm", 0.2)

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
    with open(args.flight_data, 'a') as csv_file:
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

    if timestep == 6000 and args.gust:
        sim.fdm.set_property_value("atmosphere/cosine-gust/start", 1)
        print("Wind Gust started !")

    timestep += 1
