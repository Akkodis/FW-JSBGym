#!/usr/bin/env python3
import csv
from fw_jsbgym.simulation.jsb_simulation import Simulation
import os
import argparse
import random
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel
from fw_jsbgym.agents.pid import PID
from math import pi as PI
from math import atan2
from fw_jsbgym.visualizers.visualizer import FlightGearVisualizer

# parse command line arguments
parser = argparse.ArgumentParser(description='Run JSBSim simulation.')
parser.add_argument('--fgear_viz', action='store_true', help='Enable FlightGear visualization.')
parser.add_argument('--plot_viz', action='store_true', help='Enable plot visualization.')
parser.add_argument('--aircraft_id', type=str, default='x8', help='Aircraft ID.')
parser.add_argument('--fdm_frequency', type=float, default=240.0, help='FDM frequency in Hz.')
parser.add_argument('--viz_time_factor', type=float, default=2.0, help='Visualization time factor.')
parser.add_argument('--flight_data', type=str, default='telemetry/flight_data.csv', help='Path to flight data csv file.')
parser.add_argument('--turb', action='store_true', help='Enable turbulence.')
parser.add_argument('--wind', action='store_true', help='Enable wind.')
parser.add_argument('--gust', action='store_true', help='Enable gust.')
parser.add_argument('--trim', action='store_true', help='Enable trim flight at start.')
parser.add_argument('--pid', action='store_true', help='Enable PID control.')
parser.add_argument('--outer_loop', action='store_true', help='Enable outer loop control: course, altitude, Va')
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
                enable_fgear_output=args.fgear_viz,
                enable_trim=args.trim,
                trim_point=trim_point)

if args.fgear_viz:
    fgear_viz = FlightGearVisualizer(sim)

properties = sim.fdm.query_property_catalog("atmosphere")
# sim.fdm.print_property_catalog()
# print("********PROPERTIES***********\n", properties)

# create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

fieldnames: list[str] = ['latitude', 'longitude', 'altitude', 
                         'roll', 'pitch', 'course', 
                         'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed',
                         'throttle_cmd', 'elevator_cmd', 'aileron_cmd',
                         'airspeed_ref', 'altitude_ref', 'course_ref',
                         'airspeed_err', 'altitude_err', 'course_err',
                         'pitch_err', 'roll_err',
                         'pitch_ref', 'roll_ref',
                         'aileron_pos', 'right_aileron_pos_rad', 'left_aileron_pos_rad',
                         'windspeed_north_mps', 'windspeed_east_mps', 'windspeed_down_mps',
                         'turb_north_mps', 'turb_east_mps', 'turb_down_mps'
                         ]

# create flight_data csv file with header
with open(args.flight_data, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# set seed for random number generator
rand_seed: int = random.randint(0, 1000000)
sim.fdm["simulation/randomseed"] = 1

# set constant wind sums up to 
if args.wind:
    sim.fdm["atmosphere/wind-north-fps"] = 16.26 * 3.281 # mps to fps
    sim.fdm["atmosphere/wind-east-fps"] = 16.26 * 3.281 # mps to fps
    # sim.fdm["atmosphere/wind-north-fps"] = 5.0 * 3.281 # mps to fps
    # sim.fdm["atmosphere/wind-east-fps"] = 5.0 * 3.281 # mps to fps

# set turbulences
if args.turb:
    sim.fdm["atmosphere/turb-type"] = 3 # 3 Milspec, 4 Tustin
    sim.fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = 75 # light 25 - 3 | moderate 50 - 4 | severe 75 - 6
    sim.fdm["atmosphere/turbulence/milspec/severity"] = 6

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

##### book computed gains #####
# kp_roll: float = lat_pid_gains["kp_roll"]
# ki_roll: float = lat_pid_gains["ki_roll"]
# kd_roll: float = lat_pid_gains["kd_roll"]

# kp_course: float = lat_pid_gains["kp_course"]
# ki_course: float = lat_pid_gains["ki_course"]

##### fw-airsim gains #####
# kp_roll: float = 0.20
# ki_roll: float = 0.0
# kd_roll: float = 0.089

# kp_course: float = 0.01
# ki_course: float = 0.003191

##### hand tuned gains #####
if args.outer_loop:
    # inner loop
    kp_roll: float = 0.5
    ki_roll: float = 0.0
    kd_roll: float = 0.2
    roll_pid: PID = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll,
                    dt=sim.fdm_dt, limit=x8.aileron_limit)

    # outer loop
    kp_course: float = 0.4
    ki_course: float = 0.0
    course_pid: PID = PID(kp=kp_course, ki=ki_course,
                     dt=sim.fdm_dt, limit=x8.roll_max)

else: # inner loop control only gains
    kp_roll: float = 1.0
    ki_roll: float = 0.0
    kd_roll: float = 0.5
    roll_pid: PID = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll,
                    dt=sim.fdm_dt, limit=x8.aileron_limit)


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
if args.outer_loop:
    kp_pitch: float = -19.0
    ki_pitch: float = -0.0
    kd_pitch: float = -2.0
    pitch_pid: PID = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                        dt=sim.fdm_dt, trim=trim_point, limit=x8.aileron_limit)

    kp_alt: float = 0.1
    ki_alt: float = 0.0001
    kd_alt: float = 0.0
    altitude_pid: PID = PID(kp=kp_alt, ki=ki_alt, kd=kd_alt,
                            dt=sim.fdm_dt, trim=trim_point, limit=x8.pitch_max)

    kp_airspeed: float = 1.2
    ki_airspeed: float = 0.0
    kd_airspeed: float = 0.0
    airspeed_pid: PID = PID(kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
                        dt=sim.fdm_dt, trim=trim_point, limit=x8.throttle_limit, is_throttle=True)

else: # inner loop control only gains
    kp_pitch: float = -4.0
    ki_pitch: float = -0.75
    kd_pitch: float = -0.1
    pitch_pid: PID = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=sim.fdm_dt, trim=trim_point, limit=x8.aileron_limit)

    kp_airspeed: float = 0.5
    ki_airspeed: float = 0.1
    kd_airspeed: float = 0.0
    airspeed_pid: PID = PID(kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
                        dt=sim.fdm_dt, trim=trim_point, limit=x8.throttle_limit, is_throttle=True)

# references
# outer loop refs
course_ref: float = 230 * (PI / 180)
altitude_ref: float = 2000 # ft (609.6 m)

# inner loop refs
roll_ref: float = 15 * (PI / 180) # rad
pitch_ref: float = 15 * (PI / 180) # rad
airspeed_ref: float = 34 # kts (17.49 m/s)

# initializing cmds and errors to 0 to 0
throttle_cmd = airspeed_err = altitude_err = elevator_cmd = pitch_err = \
course_err = aileron_cmd = roll_err = 0.0

# simulation loop
timestep: int = 0
while sim.run_step() and timestep < 15000:
    latitude: float = sim.fdm["position/lat-gc-deg"]
    longitude: float = sim.fdm["position/long-gc-deg"]
    altitude: float = sim.fdm["position/h-sl-ft"]

    roll: float = sim.fdm["attitude/roll-rad"]
    pitch: float = sim.fdm["attitude/pitch-rad"]
    # heading: float = sim.fdm["attitude/heading-true-rad"]
    # psi: float = sim.fdm["attitude/psi-rad"]
    # psi_gt: float = sim.fdm["flight-path/psi-gt-rad"]
    course_angle: float = (atan2(sim.fdm["velocities/v-east-fps"], sim.fdm["velocities/v-north-fps"]) + 2 * PI) % (2 * PI) # rad [0, 2PI]
    aileron_pos = sim.fdm["fcs/effective-aileron-pos"]
    right_pos_rad = sim.fdm["fcs/right-aileron-pos-rad"]
    left_pos_rad = sim.fdm["fcs/left-aileron-pos-rad"]

    # print(f"2PI = {2*PI} | psi-gt = {psi_gt}")

    roll_rate: float = sim.fdm["velocities/p-rad_sec"]
    pitch_rate: float = sim.fdm["velocities/q-rad_sec"]
    yaw_rate: float = sim.fdm["velocities/r-rad_sec"]

    # airspeed: float = sim.fdm["velocities/vc-kts"] * 1.852 # to km/h
    airspeed: float = sim.fdm["velocities/vt-fps"] / 1.688 # fps to kts

    windspeed_north_mps: float = sim.fdm["atmosphere/total-wind-north-fps"] / 3.281 # fps to mps
    windspeed_east_mps: float = sim.fdm["atmosphere/total-wind-east-fps"] / 3.281 # fps to mps
    windspeed_down_mps: float = sim.fdm["atmosphere/total-wind-down-fps"] / 3.281 # fps to mps

    turb_north_mps: float = sim.fdm["atmosphere/turb-north-fps"] / 3.281 # fps to mps
    turb_east_mps: float = sim.fdm["atmosphere/turb-east-fps"] / 3.281 # fps to mps
    turb_down_mps: float = sim.fdm["atmosphere/turb-down-fps"] / 3.281 # fps to mps

    # command longitudinal control first
    if timestep > 50 and args.pid:
        # input("Press Enter to continue...")
        # set the airspeed ref
        airspeed_pid.set_reference(airspeed_ref)
        throttle_cmd, airspeed_err = airspeed_pid.update(state=airspeed, saturate=True)
        sim.fdm["fcs/throttle-cmd-norm"] = throttle_cmd

        if args.outer_loop:
        # set the altitude ref (outer loop)
            altitude_pid.set_reference(altitude_ref)
            pitch_ref, altitude_err = altitude_pid.update(state=altitude, saturate=True)

        # set the pitch_ref (inner loop)
        pitch_pid.set_reference(pitch_ref)
        elevator_cmd, pitch_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
        sim.fdm["fcs/elevator-cmd-norm"] = elevator_cmd

        # set the ref course angle (outer loop)
        if args.outer_loop:
            course_pid.set_reference(course_ref)
            roll_ref, course_err = course_pid.update(state=course_angle, saturate=True, is_course=True)

        # set the roll_ref (inner loop)
        roll_pid.set_reference(roll_ref)
        aileron_cmd, roll_err = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)
        sim.fdm["fcs/aileron-cmd-norm"] = aileron_cmd


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
            fieldnames[4]: pitch,
            fieldnames[5]: course_angle,
            fieldnames[6]: roll_rate,
            fieldnames[7]: pitch_rate,
            fieldnames[8]: yaw_rate,
            fieldnames[9]: airspeed,
            fieldnames[10]: throttle,
            fieldnames[11]: elevator,
            fieldnames[12]: aileron,
            fieldnames[13]: airspeed_ref,
            fieldnames[14]: altitude_ref,
            fieldnames[15]: course_ref,
            fieldnames[16]: airspeed_err,
            fieldnames[17]: altitude_err,
            fieldnames[18]: course_err,
            fieldnames[19]: pitch_err,
            fieldnames[20]: roll_err,
            fieldnames[21]: pitch_ref,
            fieldnames[22]: roll_ref,
            fieldnames[23]: aileron_pos,
            fieldnames[24]: right_pos_rad,
            fieldnames[25]: left_pos_rad,
            fieldnames[26]: windspeed_north_mps,
            fieldnames[27]: windspeed_east_mps,
            fieldnames[28]: windspeed_down_mps,
            fieldnames[29]: turb_north_mps,
            fieldnames[30]: turb_east_mps,
            fieldnames[31]: turb_down_mps
        }
        csv_writer.writerow(info)

    if timestep == 6000 and args.gust:
        sim.fdm["atmosphere/cosine-gust/start"] = 1
        print("Wind Gust started !")

    timestep += 1
