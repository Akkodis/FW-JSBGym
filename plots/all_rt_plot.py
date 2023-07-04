#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from os import path
from matplotlib.animation import FuncAnimation


# setting up axis for animation
fig, ax = plt.subplots(3, 3)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# alt_plt, = ax[0, 0].plot([], [], label='altitude')
# alt_ref_plt, = ax[0, 0].plot([], [], label='altitude_ref')
# course_plt, = ax[0, 1].plot([], [], label='course')
# course_ref_plt, = ax[0, 1].plot([], [], label='course_ref')
# traj_plt, = ax[0, 2].plot([], [], [], label='Aircraft Trajectory')
# pitch_plt, = ax[1, 0].plot([], [], label='pitch')
# pitch_cmd_plt, = ax[1, 0].plot([], [], label='pitch_ref/cmd')
# roll_plt, = ax[1, 1].plot([], [], label='roll')
# roll_cmd_plt, = ax[1, 1].plot([], [], label='roll_ref/cmd')
# airspeed_plt, = ax[1, 2].plot([], [], label='airspeed')
# airspeed_ref_plt, = ax[1, 2].plot([], [], label='airspeed_ref')
# aileron_cmd_plt, = ax[2, 0].plot([], [], label='aileron_cmd')
# elevator_cmd_plt, = ax[2, 0].plot([], [], label='elevator_cmd')
# throttle_cmd_plt, = ax[2, 0].plot([], [], label='throttle_cmd')
# roll_rate_plt, = ax[2, 1].plot([], [], label='roll_rate')
# pitch_rate_plt ,= ax[2, 1].plot([], [], label='pitch_rate')
# yaw_rate_plt ,= ax[2, 1].plot([], [], label='yaw_rate')

# def init():
#     alt_plt.set_data([], [])
#     alt_ref_plt.set_data([], [])
#     course_plt.set_data([], [])
#     course_ref_plt.set_data([], [])
#     traj_plt.set_data([], [], [])
#     pitch_plt.set_data([], [])
#     pitch_cmd_plt.set_data([], [])
#     roll_plt.set_data([], [])
#     roll_cmd_plt.set_data([], [])
#     airspeed_plt.set_data([], [])
#     airspeed_ref_plt.set_data([], [])
#     aileron_cmd_plt.set_data([], [])
#     elevator_cmd_plt.set_data([], [])
#     throttle_cmd_plt.set_data([], [])
#     roll_rate_plt.set_data([], [])
#     pitch_rate_plt.set_data([], [])
#     yaw_rate_plt.set_data([], [])

#     return alt_plt, alt_ref_plt,

def animate(i, axis, args) -> None:
    data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/flight_data.csv')
    # data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/std_turb.csv')
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    roll = data['roll']
    pitch = data['pitch']
    course = data['course']

    roll_rate = data['roll_rate']
    pitch_rate = data['pitch_rate']
    yaw_rate = data['yaw_rate']

    airspeed = data['airspeed']

    throttle_cmd = data['throttle_cmd']
    elevator_cmd = data['elevator_cmd']
    aileron_cmd = data['aileron_cmd']
    pitch_cmd = data['pitch_cmd']
    roll_cmd = data['roll_cmd']

    airspeed_ref = data['airspeed_ref']
    altitude_ref = data['altitude_ref']
    course_ref = data['course_ref']

    airspeed_err = data['airspeed_err']
    altitude_err = data['altitude_err']
    course_err = data['course_err']
    pitch_err = data['pitch_err']
    roll_err = data['roll_err']

    for(dim_1) in axis:
        for(dim_2) in dim_1:
            dim_2.cla()

    num_steps = len(data.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)
    
    alt_plt ,= axis[0, 0].plot(tsteps, alt, label='altitude')
    # alt_ref_plt, = axis[0, 0].plot(tsteps, altitude_ref, label='altitude_ref')
    # axis[0, 0].plot(tsteps, altitude_err, label='altitude_err')
    axis[0, 0].set_title("altitude control (ft)")
    axis[0, 0].legend()

    course_plt, = axis[0, 1].plot(tsteps, course, label='course')
    # course_ref_plt, = axis[0, 1].plot(tsteps, course_ref, label='course_ref')
    # axis[0, 1].plot(tsteps, course_err, label='course_err')
    axis[0, 1].set_title("course control (rad)")
    axis[0, 1].legend()

    if args.scale:
        # init_zrange: list[int] = [400, 800]
        # max_boundZ: float = max(alt.max(), init_zrange[1])
        # min_boundZ: float = min(alt.min(), init_zrange[0])
        # boundZ: int = max(abs(max_boundZ), abs(min_boundZ))
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
    # pitch_cmd_plt, = axis[1, 0].plot(tsteps, pitch_cmd, label='pitch_ref/cmd')
    # axis[1, 0].plot(tsteps, pitch_err, label='pitch_err')
    axis[1, 0].set_title('pitch control (rad)')
    axis[1, 0].legend()

    roll_plt, = axis[1, 1].plot(tsteps, roll, label='roll')
    # roll_cmd_plt, = axis[1, 1].plot(tsteps, roll_cmd, label='roll_ref/cmd')
    # axis[1, 1].plot(tsteps, roll_err, label='roll_err')
    axis[1, 1].set_title('roll control (rad)')
    axis[1, 1].legend()

    airspeed_plt, = axis[1, 2].plot(tsteps, airspeed, label='airspeed')
    # airspeed_ref_plt, = axis[1, 2].plot(tsteps, airspeed_ref, label='airspeed_ref')
    # axis[1, 2].plot(tsteps, airspeed_err, label='airspeed_err')
    axis[1, 2].set_title('airspeed control (kts)')
    axis[1, 2].legend()

    aileron_cmd_plt, = axis[2, 0].plot(tsteps, aileron_cmd, label='aileron_cmd')
    elevator_cmd_plt, = axis[2, 0].plot(tsteps, elevator_cmd, label='elevator_cmd')
    throttle_cmd_plt, = axis[2, 0].plot(tsteps, throttle_cmd, label='throttle_cmd')
    axis[2, 0].set_title('commands')
    axis[2, 0].legend()

    roll_rate_plt, = axis[2, 1].plot(tsteps, roll_rate, label='roll_rate')
    pitch_rate_plt, = axis[2, 1].plot(tsteps, pitch_rate, label='pitch_rate')
    yaw_rate_plt, = axis[2, 1].plot(tsteps, yaw_rate, label='yaw_rate')
    axis[2, 1].set_title('angular velocities (rad/s)')
    axis[2, 1].legend()

    ax[2, 2].set_axis_off()

    # return alt_plt, alt_ref_plt,
    # return alt_plt, alt_ref_plt, course_plt, course_ref_plt, traj_plt, pitch_plt, pitch_cmd_plt, \
    #        roll_plt, roll_cmd_plt, airspeed_plt, airspeed_ref_plt, aileron_cmd_plt, elevator_cmd_plt, throttle_cmd_plt, \
    #        roll_rate_plt, pitch_rate_plt, yaw_rate_plt,

# parse command line arguments
parser = ArgumentParser(description='Plotting Telemetry Data')
parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
args: Namespace = parser.parse_args()



ax[0, 2].remove()
ax[0, 2] = fig.add_subplot(3, 3, 3, projection='3d')
# ax[0, 2].set_aspect('equalxy', 'box')
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, args, ), interval=50, blit=False)

plt.show()
