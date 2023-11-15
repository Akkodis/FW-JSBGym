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
    data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../telemetry/eval_telemetry.csv')

    lat = data[prp.lat_gc_deg.get_legal_name()]
    lon = data[prp.lng_gc_deg.get_legal_name()]
    alt = data[prp.altitude_sl_m.get_legal_name()]

    roll = data[prp.roll_rad.get_legal_name()]
    pitch = data[prp.pitch_rad.get_legal_name()]
    heading = data[prp.heading_rad.get_legal_name()]

    roll_rate = data[prp.p_radps.get_legal_name()]
    pitch_rate = data[prp.q_radps.get_legal_name()]
    yaw_rate = data[prp.r_radps.get_legal_name()]

    airspeed = data[prp.airspeed_mps.get_legal_name()]

    throttle_cmd = data[prp.throttle_cmd.get_legal_name()]
    elevator_cmd = data[prp.elevator_cmd.get_legal_name()]
    aileron_cmd = data[prp.aileron_cmd.get_legal_name()]

    airspeed_ref = data[prp.target_airspeed_mps.get_legal_name()]
    roll_ref = data[prp.target_roll_rad.get_legal_name()]
    pitch_ref = data[prp.target_pitch_rad.get_legal_name()]

    r_total = data[prp.reward_total.get_legal_name()]
    r_roll = data[prp.reward_roll.get_legal_name()]
    r_pitch = data[prp.reward_pitch.get_legal_name()]
    r_airspeed = data[prp.reward_airspeed.get_legal_name()]
    r_actvar = data[prp.reward_actvar.get_legal_name()]


    for(dim_1) in axis:
        for(dim_2) in dim_1:
            dim_2.cla()

    num_steps = len(data.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)
    
    alt_plt ,= axis[0, 0].plot(tsteps, alt, label='altitude')
    axis[0, 0].set_title("altitude control [m]")
    axis[0, 0].legend()

    course_plt, = axis[0, 1].plot(tsteps, heading, label='course')
    axis[0, 1].set_title("Heading (psi) control [rad]")
    axis[0, 1].legend()

    if args.scale:
        axis[0, 2].set_zlim(alt.iat[-1]-200, alt.iat[-1]+200)
        max_bound2D: float = max(lat.max(), lon.max())
        min_bound2D: float = min(lat.min(), lon.min())
        bound2D: float = max(abs(max_bound2D), abs(min_bound2D))
        axis[0, 2].set_xlim(-bound2D, bound2D)
        axis[0, 2].set_ylim(-bound2D, bound2D)

    traj_plt, = axis[0, 2].plot(lon, lat, alt, label='Aircraft Trajectory')
    axis[0, 2].legend()
    # plt.tight_layout()

    pitch_plt, = axis[1, 0].plot(tsteps, pitch, label='pitch')
    axis[1, 0].set_title('pitch control [rad]')
    axis[1, 0].legend()

    roll_plt, = axis[1, 1].plot(tsteps, roll, label='roll')
    axis[1, 1].set_title('roll control [rad]')
    axis[1, 1].legend()

    airspeed_plt, = axis[1, 2].plot(tsteps, airspeed, label='airspeed')
    airspeed_ref_plt, = axis[1, 2].plot(tsteps, airspeed_ref, label='airspeed_ref')
    axis[1, 2].set_title('airspeed control [m/s]')
    axis[1, 2].legend()

    aileron_cmd_plt, = axis[2, 0].plot(tsteps, aileron_cmd, label='aileron_cmd')
    elevator_cmd_plt, = axis[2, 0].plot(tsteps, elevator_cmd, label='elevator_cmd')
    throttle_cmd_plt, = axis[2, 0].plot(tsteps, throttle_cmd, label='throttle_cmd')
    axis[2, 0].set_title('commands')
    axis[2, 0].legend()

    roll_rate_plt, = axis[2, 1].plot(tsteps, roll_rate, label='roll_rate')
    pitch_rate_plt, = axis[2, 1].plot(tsteps, pitch_rate, label='pitch_rate')
    yaw_rate_plt, = axis[2, 1].plot(tsteps, yaw_rate, label='yaw_rate')
    axis[2, 1].set_title('angular velocities [rad/s]')
    axis[2, 1].legend()
    
    r_total_plt, = axis[2, 2].plot(tsteps, r_total, label='r_total')
    r_roll_plt, = axis[2, 2].plot(tsteps, r_roll, label='r_roll')
    r_pitch_plt, = axis[2, 2].plot(tsteps, r_pitch, label='r_pitch')
    r_airspeed_plt, = axis[2, 2].plot(tsteps, r_airspeed, label='r_airspeed')
    r_actvar_plt, = axis[2, 2].plot(tsteps, r_actvar, label='r_actvar')
    axis[2, 2].set_title('rewards')
    axis[2, 2].legend()
    # ax[2, 2].set_axis_off()


# parse command line arguments
parser = ArgumentParser(description='Plotting Telemetry Data')
parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
parser.add_argument('--csv', type=str, default='data/test_gym_flight_data.csv', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
parser.add_argument('--fullscreen', action='store_true', help='True: fullscreen, False: windowed')
args: Namespace = parser.parse_args()

# setting up axis for animation
fig, ax = plt.subplots(3, 3)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
if args.fullscreen:
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

ax[0, 2].remove()
ax[0, 2] = fig.add_subplot(3, 3, 3, projection='3d')
# ax[0, 2].set_aspect('equalxy', 'box')
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, args, ), interval=50, blit=False)
print("Animation plot started...", file=sys.stderr)

plt.show()
