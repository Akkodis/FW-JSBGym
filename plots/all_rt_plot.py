#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from os import path
from matplotlib.animation import FuncAnimation


def animate_states(i, axis, args) -> None:
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

    axis[0, 0].plot(tsteps, alt, label='altitude')
    axis[0, 0].plot(tsteps, altitude_ref, label='altitude_ref')
    # axis[0, 0].plot(tsteps, altitude_err, label='altitude_err')
    axis[0, 0].set_title("altitude control (ft)")
    axis[0, 0].legend()

    axis[0, 1].plot(tsteps, course, label='course')
    axis[0, 1].plot(tsteps, course_ref, label='course_ref')
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

    axis[0, 2].plot(lon, lat, alt, label='Aircraft Trajectory')
    axis[0, 2].legend()
    plt.tight_layout()

    axis[1, 0].plot(tsteps, pitch, label='pitch')
    axis[1, 0].plot(tsteps, pitch_cmd, label='pitch_ref/cmd')
    # axis[1, 0].plot(tsteps, pitch_err, label='pitch_err')
    axis[1, 0].set_title('pitch control (rad)')
    axis[1, 0].legend()

    axis[1, 1].plot(tsteps, roll, label='roll')
    axis[1, 1].plot(tsteps, roll_cmd, label='roll_ref/cmd')
    # axis[1, 1].plot(tsteps, roll_err, label='roll_err')
    axis[1, 1].set_title('roll control (rad)')
    axis[1, 1].legend()

    axis[1, 2].plot(tsteps, airspeed, label='airspeed')
    axis[1, 2].plot(tsteps, airspeed_ref, label='airspeed_ref')
    # axis[1, 2].plot(tsteps, airspeed_err, label='airspeed_err')
    axis[1, 2].set_title('airspeed control (kts)')
    axis[1, 2].legend()

    axis[2, 0].plot(tsteps, aileron_cmd, label='aileron_cmd')
    axis[2, 0].plot(tsteps, elevator_cmd, label='elevator_cmd')
    axis[2, 0].plot(tsteps, throttle_cmd, label='throttle_cmd')
    axis[2, 0].set_title('commands')
    axis[2, 0].legend()

    axis[2, 1].plot(tsteps, roll_rate, label='roll_rate')
    axis[2, 1].plot(tsteps, pitch_rate, label='pitch_rate')
    axis[2, 1].plot(tsteps, yaw_rate, label='yaw_rate')
    axis[2, 1].set_title('angular velocities (rad/s)')
    axis[2, 1].legend()


# parse command line arguments
parser = ArgumentParser(description='Run JSBSim simulation.')
parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
args: Namespace = parser.parse_args()

# figure animation for states
fig_s, ax_s = plt.subplots(3, 3)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
ax_s[0, 2].remove()
ax_s[0, 2] = fig_s.add_subplot(3, 3, 3, projection='3d')
# ax[0, 2].set_aspect('equalxy', 'box')
ani = FuncAnimation(plt.gcf(), animate_states, fargs=(ax_s, args, ), interval=50)

plt.show()
