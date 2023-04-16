#!/usr/bin/env python3
import csv
from simulation.jsb_simulation import Simulation
import os
import argparse
import random
from trim.trim_point import TrimPoint
from models.aerodynamics import AeroModel
from agents.pid import PID
from math import pi as PI
from math import atan2

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
parser.add_argument('--trim', action='store_true', help='Enable trim flight at start.')
args: argparse.Namespace = parser.parse_args()

# if trim is enabled, construct TrimPoint object
if args.trim:
    trim_point: TrimPoint = TrimPoint(aircraft_id=args.aircraft_id)
else:
    trim_point: TrimPoint = None

# create a simulation object accordingly
sim: Simulation = Simulation(fdm_frequency=args.fdm_frequency, # going up to 240Hz solves some spin instability issues -> NaNs
                aircraft_id=args.aircraft_id,
                viz_time_factor=args.viz_time_factor,
                enable_fgear_viz=args.fgear_viz,
                enable_trim=args.trim,
                trim_point=trim_point)

properties = sim.fdm.query_property_catalog("atmosphere")
# sim.fdm.print_property_catalog()
# print("********PROPERTIES***********\n", properties)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

fieldnames: list[str] = ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']
# fieldnames: list[str] = ['latitude', 'longitude', 'altitude', 'roll', 'course', 'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed']

# create flight_data csv file with header
with open(args.flight_data, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# set seed for random number generator
rand_seed: int = random.randint(0, 1000000)
sim.fdm["simulation/randomseed"] = rand_seed

# set wind
if args.wind:
    # sim.fdm["atmosphere/wind-north-fps"] = 100
    sim.fdm["atmosphere/wind-east-fps"] = 40

# set turbulences
if args.turb:
    sim.fdm["atmosphere/turb-type"] = 4 # Tustin turbulence type
    sim.fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = 30
    sim.fdm["atmosphere/turbulence/milspec/severity"] = 3

# set wind gust
if args.gust:
    sim.fdm["atmosphere/cosine-gust/startup-duration-sec"] = 2
    sim.fdm["atmosphere/cosine-gust/steady-duration-sec"] = 2
    sim.fdm["atmosphere/cosine-gust/end-duration-sec"] = 2
    sim.fdm["atmosphere/cosine-gust/magnitude-ft_sec"] = 40
    sim.fdm["atmosphere/cosine-gust/frame"] = 2
    sim.fdm["atmosphere/cosine-gust/X-velocity-ft_sec"] = 1
    sim.fdm["atmosphere/cosine-gust/Y-velocity-ft_sec"] = 0
    sim.fdm["atmosphere/cosine-gust/Z-velocity-ft_sec"] = 0

# if trim is enabled, set the according flight controls to maintain trimmed flight
if args.trim:
    sim.fdm["fcs/throttle-cmd-norm"] = trim_point.throttle
    sim.fdm["fcs/aileron-cmd-norm"] = trim_point.aileron
    sim.fdm["fcs/elevator-cmd-norm"] = trim_point.elevator

# create the aerodynamics model
aero_model: AeroModel = AeroModel()

# compute the lateral PID gains
lat_pid_gains: dict[str, float]
lat_resp_times: dict[str, float]
lat_pid_gains, lat_resp_times = aero_model.compute_lat_pid_gains()

# create lateral PID controller
# roll PID (inner loop)
roll_pid: PID = PID(kp=lat_pid_gains["kp_roll"], ki=lat_pid_gains["ki_roll"], kd=lat_pid_gains["kd_roll"],
                    dt=sim.fdm_dt, limit=aero_model.aileron_limit)

# course angle PID (outer loop)
course_pid: PID = PID(kp=lat_pid_gains["kp_course"], ki=lat_pid_gains["ki_course"],
                      dt=sim.fdm_dt, limit=aero_model.roll_max)

# compute the longitudinal PID gains
long_pid_gains: dict[str, float]
long_resp_times: dict[str, float]
long_pid_gains, long_resp_times, _ = aero_model.compute_long_pid_gains()
pitch_pid: PID = PID(kp=1.0, ki=0, kd=0.03,
                     dt=sim.fdm_dt, limit=aero_model.aileron_limit)
altitude_pid: PID = PID(kp=0.1, ki=0.6,
                        dt=sim.fdm_dt, limit=aero_model.pitch_max)
airspeed_pid: PID = PID(kp=1.0, ki=0.1, kd=0, dt=sim.fdm_dt, limit=aero_model.throttle_limit, is_throttle=True)
# pitch_pid: PID = PID(kp=long_pid_gains["kp_pitch"], ki=0, kd=long_pid_gains["kd_pitch"],
#                      dt=sim.fdm_dt, limit=aero_model.aileron_limit)
# altitude_pid: PID = PID(kp=long_pid_gains["kp_h"], ki=long_pid_gains["ki_h"],
#                         dt=sim.fdm_dt, limit=aero_model.pitch_max)


course_ref: float = 10.0 * (PI / 180)
altitude_ref: float = 600
airspeed_ref: float = 62 / 1.852 # km/h to kts
# simulation loop
timestep: int = 0
while sim.run_step() and timestep < 20000:
    latitude: float = sim.fdm["position/lat-gc-deg"]
    longitude: float = sim.fdm["position/long-gc-deg"]
    altitude: float = sim.fdm["position/h-sl-meters"]

    roll: float = sim.fdm["attitude/roll-rad"]
    pitch: float = sim.fdm["attitude/pitch-rad"]
    heading: float = sim.fdm["attitude/heading-true-rad"]
    psi: float = sim.fdm["attitude/psi-rad"]
    psi_gt: float = sim.fdm["flight-path/psi-gt-rad"]

    # print(f"2PI = {2*PI} | psi-gt = {psi_gt}")

    roll_rate: float = sim.fdm["velocities/p-rad_sec"]
    pitch_rate: float = sim.fdm["velocities/q-rad_sec"]
    yaw_rate: float = sim.fdm["velocities/r-rad_sec"]

    airspeed: float = sim.fdm["velocities/vc-kts"] * 1.852 # to km/h
    airspeed: float = sim.fdm["velocities/vc-kts"] # kts

    if timestep > 4000:

        # # set the airspeed ref
        # throttle_cmd: float
        # throttle_err: float
        # airspeed_pid.set_reference(airspeed_ref)
        # throttle_cmd, throttle_err = airspeed_pid.update(state=airspeed, saturate=True)
        # sim.fdm["fcs/throttle-cmd-norm"] = throttle_cmd
        # print("throttle_err = ", throttle_err)


        # set the altitude ref
        pitch_cmd: float
        pitch_err: float
        altitude_pid.set_reference(altitude_ref)
        pitch_cmd, pitch_err = altitude_pid.update(state=altitude)

        elevator_cmd: float
        elevator_err: float
        pitch_pid.set_reference(pitch_cmd)
        elevator_cmd, elevator_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
        sim.fdm["fcs/elevator-cmd-norm"] = elevator_cmd
        print(f"h = {altitude}")
        print(f"de = {elevator_cmd}")

        # set the ref course angle to be a 90° right turn
        # roll_cmd: float
        # aileron_cmd: float
        # course_pid.set_reference(course_ref)
        # course_angle: float = atan2(sim.fdm["velocities/v-east-fps"], sim.fdm["velocities/v-north-fps"])
        # course_cmd, _ = course_pid.update(state=course_angle, normalize=False) # don't normalize it between -1 and 1
        # roll_pid.set_reference(roll_cmd)
        # aileron_cmd = roll_pid.update(state=sim.fdm["attitude/roll-rad"], state_dot=sim.fdm["velocities/p-rad_sec"], normalize=True)
        # print(f"aileron_cmd: {aileron_cmd} | roll_cmd: {roll_cmd}")
        # sim.fdm["fcs/aileron-cmd-rad"]

    
    # write flight data to csv
    with open(args.flight_data, 'a') as csv_file:
        csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info: dict[str, float] = {
            fieldnames[0]: latitude,
            fieldnames[1]: longitude,
            fieldnames[2]: altitude,
            fieldnames[3]: roll,
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

    if timestep == 6000 and args.gust:
        sim.fdm["atmosphere/cosine-gust/start"] = 1
        print("Wind Gust started !")

    timestep += 1
