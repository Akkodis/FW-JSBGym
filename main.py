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
from simple_pid import PID as SPID

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

fieldnames: list[str] = ['latitude', 'longitude', 'altitude', 
                         'roll', 'pitch', 'yaw', 
                         'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed',
                         'throttle_cmd', 'elevator_cmd', 'aileron_cmd']
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
x8: AeroModel = AeroModel()

# compute the lateral PID gains
lat_pid_gains: dict[str, float]
lat_resp_times: dict[str, float]
lat_pid_gains, lat_resp_times = x8.compute_lat_pid_gains()

# create lateral PID controller
# roll PID (inner loop)
roll_pid: PID = PID(kp=lat_pid_gains["kp_roll"], ki=lat_pid_gains["ki_roll"], kd=lat_pid_gains["kd_roll"],
                    limit=x8.aileron_limit)

# course angle PID (outer loop)
course_pid: PID = PID(kp=lat_pid_gains["kp_course"], ki=lat_pid_gains["ki_course"],
                      limit=x8.roll_max)

# compute the longitudinal PID gains
long_pid_gains: dict[str, float]
long_resp_times: dict[str, float]
long_pid_gains, long_resp_times, _ = x8.compute_long_pid_gains()

# kp gains for pitch must be negative, because a negative elevator deflection is required to increase pitch
##### book computed gains #######
# kp_pitch: float = long_pid_gains["kp_pitch"] # -1.0
# ki_pitch: float = 0.0
# kd_pitch: float = long_pid_gains["kd_pitch"] # -0.388

# kp_alt: float = long_pid_gains["kp_h"] #0.479
# ki_alt: float = long_pid_gains["ki_h"] # 0.322
# kd_alt: float = 0.0

# kp_airspeed: float = long_pid_gains["kp_vth"] # 0.604
# ki_airspeed: float = long_pid_gains["ki_vth"] # 0.678
# kd_airspeed: float = 0.0

##### fw-airsim gains #######
# kp_pitch: float = -1.0
# ki_pitch: float = -0.0
# kd_pitch: float = -0.03

# kp_alt: float = 0.1
# # ki_alt: float = 0.6
# ki_alt: float = 0.0
# kd_alt: float = 0.0

# kp_airspeed: float = 1.0
# # ki_airspeed: float = 0.035
# ki_airspeed: float = 0.0
# kd_airspeed: float = 0.0

##### my tuned gains #######
kp_pitch: float = -1.0
ki_pitch: float = -0.0
kd_pitch: float = -0.0

kp_alt: float = 0.015
ki_alt: float = 0.0005
kd_alt: float = 0.0

kp_airspeed: float = 0.9
ki_airspeed: float = 0.001
kd_airspeed: float = 0.0

pitch_pid: PID = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=sim.fdm_dt, trim=trim_point, limit=x8.aileron_limit)
altitude_pid: PID = PID(kp=kp_alt, ki=ki_alt, kd=kd_alt,
                        dt=sim.fdm_dt, trim=trim_point, limit=x8.pitch_max)
airspeed_pid: PID = PID(kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
                        dt=sim.fdm_dt, trim=trim_point, limit=x8.throttle_limit, is_throttle=True)

# references
course_ref: float = 10.0 * (PI / 180)
altitude_ref: float = 2000 # ft
airspeed_ref: float = 34 # kts

# simulation loop
timestep: int = 0
while sim.run_step() and timestep < 20000:
    latitude: float = sim.fdm["position/lat-gc-deg"]
    longitude: float = sim.fdm["position/long-gc-deg"]
    altitude: float = sim.fdm["position/h-sl-ft"]

    roll: float = sim.fdm["attitude/roll-rad"]
    pitch: float = sim.fdm["attitude/pitch-rad"]
    heading: float = sim.fdm["attitude/heading-true-rad"]
    psi: float = sim.fdm["attitude/psi-rad"]
    psi_gt: float = sim.fdm["flight-path/psi-gt-rad"]

    # print(f"2PI = {2*PI} | psi-gt = {psi_gt}")

    roll_rate: float = sim.fdm["velocities/p-rad_sec"]
    pitch_rate: float = sim.fdm["velocities/q-rad_sec"]
    yaw_rate: float = sim.fdm["velocities/r-rad_sec"]

    # airspeed: float = sim.fdm["velocities/vc-kts"] * 1.852 # to km/h
    airspeed: float = sim.fdm["velocities/vt-fps"] * 0.5925 # fps to kts

    print(f"h = {altitude}")
    print(f"Va = {airspeed}")

    if timestep > 2000:
        # input("Press Enter to continue...")
        # set the airspeed ref
        throttle_cmd: float
        airspeed_err: float
        airspeed_pid.set_reference(airspeed_ref)
        throttle_cmd, airspeed_err = airspeed_pid.update(state=airspeed, saturate=True)
        sim.fdm["fcs/throttle-cmd-norm"] = throttle_cmd
        print("airspeed_err = ", airspeed_err)
        print("dt = ", sim.fdm["fcs/throttle-cmd-norm"])

        # set the altitude ref
        pitch_cmd: float
        altitude_err: float
        altitude_pid.set_reference(altitude_ref)
        pitch_cmd, altitude_err = altitude_pid.update(state=altitude, saturate=True)
        print(f"alt_err = {altitude_err}")
        print(f"pitch_cmd = {pitch_cmd}")

        elevator_cmd: float
        pitch_err: float
        pitch_pid.set_reference(pitch_cmd)
        elevator_cmd, pitch_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
        sim.fdm["fcs/elevator-cmd-norm"] = elevator_cmd
        print(f"pitch_err = {pitch_err}")
        print("de = ", sim.fdm["fcs/elevator-cmd-norm"])

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

        ### SPID usage
        # err_va: float = airspeed_ref - airspeed
        # airspeed_controller = SPID(kp_airspeed, ki_airspeed, kd_airspeed)
        # throttle_cmd_2 = airspeed_controller(-err_va)
        # if throttle_cmd_2 > 1:
        #     throttle_cmd_2 = 1
        # if throttle_cmd_2 < 0:
        #     throttle_cmd_2 = 0
        # sim.fdm["fcs/throttle-cmd-norm"] = throttle_cmd_2

        # err_alt: float = altitude_ref - altitude
        # altitude_controller = SPID(kp_alt, ki_alt, kd_alt)
        # pitch_cmd_2 = altitude_controller(-err_alt)
        # p, i, d = altitude_controller.components
        # if pitch_cmd_2 < -10 * (PI / 180):
        #     pitch_cmd_2 = -10 * (PI / 180)
        # if pitch_cmd_2 > 15 * (PI / 180):
        #     pitch_cmd_2 = 15 * (PI / 180)

        # err_pitch: float = pitch_cmd_2 - pitch
        # pitch_controller = SPID(kp_pitch, ki_pitch, 0.0)
        # elevator_cmd_pi = pitch_controller(err_pitch)
        # rate_controller = SPID(kd_pitch, 0.0, 0.0)
        # elevator_cmd_d = rate_controller(pitch_rate)
        # elevator_cmd_2 = elevator_cmd_pi + elevator_cmd_d
        # sim.fdm["fcs/elevator-cmd-norm"] = elevator_cmd_2

    # controls
    throttle: float = sim.fdm["fcs/throttle-cmd-norm"]
    aileron: float = sim.fdm["fcs/aileron-cmd-norm"]
    elevator: float = sim.fdm["fcs/elevator-cmd-norm"]

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
            fieldnames[9]: airspeed,
            fieldnames[10]: throttle,
            fieldnames[11]: elevator,
            fieldnames[12]: aileron
        }
        csv_writer.writerow(info)

    if timestep == 6000 and args.gust:
        sim.fdm["atmosphere/cosine-gust/start"] = 1
        print("Wind Gust started !")

    timestep += 1
