#!/usr/bin/env python3
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import numpy as np

def animate(i, axis) -> None:
    data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/flight_data.csv')
    # data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/std_turb.csv')
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    roll = data['roll']
    pitch = data['pitch']
    yaw = data['yaw']

    # course = data['course']
    # heading = data['heading-true-rad']
    # psi = data['psi-rad']
    # psi_gt = data['psi-gt-rad']

    roll_rate = data['roll_rate']
    pitch_rate = data['pitch_rate']
    yaw_rate = data['yaw_rate']

    airspeed = data['airspeed']

    for(dim_1) in axis:
        for(dim_2) in dim_1:
            dim_2.cla()

    num_steps = len(data.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)

    axis[0, 0].plot(tsteps, roll, label='roll')
    axis[0, 0].plot(tsteps, pitch, label='pitch')
    axis[0, 0].plot(tsteps, yaw, label='yaw')
    axis[0, 0].set_title("attitude angles (rad)")
    axis[0, 0].legend()

    # axis[0, 0].plot(tsteps, roll, label='roll')
    # axis[0, 0].plot(tsteps, course, label='course')
    # axis[0, 0].plot(tsteps, roll_rate, label='roll_rate')
    # axis[0, 0].set_title("course angles ?")
    # axis[0, 0].legend()

    axis[0, 1].plot(tsteps, roll_rate, label='roll_rate')
    axis[0, 1].plot(tsteps, pitch_rate, label='pitch_rate')
    axis[0, 1].plot(tsteps, yaw_rate, label='yaw_rate')
    axis[0, 1].set_title("angular velocities (rad/s)")
    axis[0, 1].legend()

    axis[1, 0].plot(tsteps, airspeed, label='airspeed')
    axis[1, 0].set_title("airspeed (km/h)")
    axis[1, 0].legend()

    axis[1, 1].plot(lon, lat, alt, label='Aircraft Trajectory')
    axis[1, 1].legend()
    plt.tight_layout()


fig, ax = plt.subplots(2, 2)
ax[1, 1].remove()
ax[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
ax[1, 1].set_aspect('equal')
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, ), interval=50)
plt.show()
