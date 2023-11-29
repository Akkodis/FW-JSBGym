#!/usr/bin/env python3
import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from matplotlib.animation import FuncAnimation
from jsbgym.utils import jsbsim_properties as prp


def animate(i, axis, args) -> None:
    df = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../{args.tele_file}')
    nan_arr = np.empty(df.index.size)
    nan_arr.fill(np.nan)

    lat = df.get(prp.lat_gc_deg.get_legal_name(), default=nan_arr)
    lon = df.get(prp.lng_gc_deg.get_legal_name(), default=nan_arr)
    alt = df.get(prp.altitude_sl_m.get_legal_name(), default=nan_arr)

    roll = df.get(prp.roll_rad.get_legal_name(), default=nan_arr)
    pitch = df.get(prp.pitch_rad.get_legal_name(), default=nan_arr)
    heading = df.get(prp.heading_rad.get_legal_name(), default=nan_arr)

    roll_rate = df.get(prp.p_radps.get_legal_name(), default=nan_arr)
    pitch_rate = df.get(prp.q_radps.get_legal_name(), default=nan_arr)
    yaw_rate = df.get(prp.r_radps.get_legal_name(), default=nan_arr)

    airspeed = df.get(prp.airspeed_kph.get_legal_name(), default=nan_arr)

    windspeed_n_kph = df.get(prp.windspeed_north_kph.get_legal_name(), default=nan_arr)
    windspeed_e_kph = df.get(prp.windspeed_east_kph.get_legal_name(), default=nan_arr)
    windspeed_d_kph = df.get(prp.windspeed_down_kph.get_legal_name(), default=nan_arr)

    turb_n_kph = df.get(prp.turb_north_kph.get_legal_name(), default=nan_arr)
    turb_e_kph = df.get(prp.turb_east_kph.get_legal_name(), default=nan_arr)
    turb_d_kph = df.get(prp.turb_down_kph.get_legal_name(), default=nan_arr)

    throttle_cmd = df.get(prp.throttle_cmd.get_legal_name(), default=nan_arr)
    elevator_cmd = df.get(prp.elevator_cmd.get_legal_name(), default=nan_arr)
    aileron_cmd = df.get(prp.aileron_cmd.get_legal_name(), default=nan_arr)

    airspeed_ref = df.get(prp.target_airspeed_mps.get_legal_name(), default=nan_arr)
    roll_ref = df.get(prp.target_roll_rad.get_legal_name(), default=nan_arr)
    pitch_ref = df.get(prp.target_pitch_rad.get_legal_name(), default=nan_arr)

    r_total = df.get(prp.reward_total.get_legal_name(), default=nan_arr)
    r_roll = df.get(prp.reward_roll.get_legal_name(), default=nan_arr)
    r_pitch = df.get(prp.reward_pitch.get_legal_name(), default=nan_arr)
    r_airspeed = df.get(prp.reward_airspeed.get_legal_name(), default=nan_arr)
    r_actvar = df.get(prp.reward_actvar.get_legal_name(), default=nan_arr)


    for(dim_1) in axis:
        for(dim_2) in dim_1:
            dim_2.cla()

    num_steps = len(df.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)
    
    axis[0, 0].plot(tsteps, alt, label='altitude' if not np.isnan(np.sum(alt)) else '')
    axis[0, 0].set_title("altitude control [m]")
    axis[0, 0].legend()
    axis[0, 0].grid()

    axis[0, 1].plot(tsteps, heading, label='course' if not np.isnan(np.sum(heading)) else '')
    axis[0, 1].set_title("Heading (psi) control [rad]")
    axis[0, 1].legend()
    axis[0, 1].grid()

    axis[0, 2].plot(tsteps, airspeed, label='airspeed' if not np.isnan(np.sum(airspeed)) else '')
    axis[0, 2].plot(tsteps, airspeed_ref, color='r', linestyle='--', label='airspeed_ref' if not np.isnan(np.sum(airspeed_ref)) else '')
    axis[0, 2].set_title('airspeed control [km/h]')
    axis[0, 2].legend()
    axis[0, 2].grid()

    # wait for the telemetry file to be filled with some data so that the plotter doesn't crash when computing scale bounds
    # if args.scale and df.index.size > 0:
    #     axis[0, 2].set_zlim(alt.min() - 50, alt.max() + 50)
    #     axis[0, 2].set_xlim(lon.min(), lon.max())
    #     axis[0, 2].set_ylim(lat.min(), lat.max())

    # axis[0, 2].plot(lon, lat, alt, label='Aircraft Trajectory')
    # axis[0, 2].legend()

    axis[1, 0].plot(tsteps, pitch, label='pitch' if not np.isnan(np.sum(pitch)) else '')
    axis[1, 0].plot(tsteps, pitch_ref, color='r', linestyle='--', label='pitch_ref' if not np.isnan(np.sum(pitch_ref)) else '')
    axis[1, 0].set_title('pitch control [rad]')
    axis[1, 0].legend()
    axis[1, 0].grid()

    axis[1, 1].plot(tsteps, roll, label='roll' if not np.isnan(np.sum(roll)) else '')
    axis[1, 1].plot(tsteps, roll_ref, color='r', linestyle='--', label='roll_ref' if not np.isnan(np.sum(roll_ref)) else '')
    axis[1, 1].set_title('roll control [rad]')
    axis[1, 1].legend()
    axis[1, 1].grid()

    axis[1, 2].plot(tsteps, windspeed_n_kph, label='north' if not np.isnan(np.sum(windspeed_n_kph)) else '')
    axis[1, 2].plot(tsteps, windspeed_e_kph, label='east' if not np.isnan(np.sum(windspeed_e_kph)) else '')
    axis[1, 2].plot(tsteps, windspeed_d_kph, label='down' if not np.isnan(np.sum(windspeed_d_kph)) else '')
    axis[1, 2].set_title('windspeed [km/h]')
    axis[1, 2].legend()
    axis[1, 2].grid()

    axis[2, 0].plot(tsteps, aileron_cmd, label='aileron_cmd' if not np.isnan(np.sum(aileron_cmd)) else '')
    axis[2, 0].plot(tsteps, elevator_cmd, label='elevator_cmd' if not np.isnan(np.sum(elevator_cmd)) else '')
    axis[2, 0].plot(tsteps, throttle_cmd, label='throttle_cmd' if not np.isnan(np.sum(throttle_cmd)) else '')
    axis[2, 0].set_title('commands')
    axis[2, 0].legend()
    axis[2, 0].grid()

    axis[2, 1].plot(tsteps, roll_rate, label='roll_rate' if not np.isnan(np.sum(roll_rate)) else '')
    axis[2, 1].plot(tsteps, pitch_rate, label='pitch_rate' if not np.isnan(np.sum(pitch_rate)) else '')
    axis[2, 1].plot(tsteps, yaw_rate, label='yaw_rate' if not np.isnan(np.sum(yaw_rate)) else '')
    axis[2, 1].set_title('angular velocities [rad/s]')
    axis[2, 1].legend()
    axis[2, 1].grid()

    axis[2, 2].plot(tsteps, r_total, label='r_total' if not np.isnan(np.sum(r_total)) else '')
    axis[2, 2].plot(tsteps, r_roll, label='r_roll' if not np.isnan(np.sum(r_roll)) else '')
    axis[2, 2].plot(tsteps, r_pitch, label='r_pitch' if not np.isnan(np.sum(r_pitch)) else '')
    axis[2, 2].plot(tsteps, r_airspeed, label='r_airspeed' if not np.isnan(np.sum(r_airspeed)) else '')
    axis[2, 2].plot(tsteps, r_actvar, label='r_actvar' if not np.isnan(np.sum(r_actvar)) else '')
    axis[2, 2].set_title('rewards')
    axis[2, 2].legend()
    axis[2, 2].grid()


# parse command line arguments
parser = ArgumentParser(description='Plotting Telemetry Data')
parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
parser.add_argument('--tele-file', type=str, required=True, help='Telemetry csv file from where to read and plot data from')
parser.add_argument('--fullscreen', action='store_true', help='True: fullscreen, False: windowed')
args: Namespace = parser.parse_args()

# setting up axis for animation
fig, ax = plt.subplots(3, 3)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
if args.fullscreen:
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

# Setting 3D subplot for trajectory plot
# ax[0, 2].remove()
# ax[0, 2] = fig.add_subplot(3, 3, 3, projection='3d')

# starting animation
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, args, ), interval=50, blit=False)
print("Animation plot started...", file=sys.stderr)
plt.show()
