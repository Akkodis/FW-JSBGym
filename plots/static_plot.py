#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path


def plot(data) -> None:
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    roll = data['roll']
    pitch = data['pitch']
    yaw = data['yaw']

    roll_rate = data['roll_rate']
    pitch_rate = data['pitch_rate']
    yaw_rate = data['yaw_rate']

    airspeed = data['airspeed']

    fig, ax = plt.subplots(2,2)
    num_steps = len(data.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)

    ax[0,0].plot(tsteps, roll, label='roll')
    ax[0,0].plot(tsteps, pitch, label='pitch')
    ax[0,0].plot(tsteps, yaw, label='yaw')
    ax[0,0].set_title("attitude angles")
    ax[0,0].legend()

    ax[0,1].plot(tsteps, roll_rate, label='roll_rate')
    ax[0,1].plot(tsteps, pitch_rate, label='pitch_rate')
    ax[0,1].plot(tsteps, yaw_rate, label='yaw_rate')
    ax[0,1].set_title("angular velocities")
    ax[0,1].legend()

    ax[1,0].plot(tsteps, airspeed, label='airspeed')
    ax[1,0].set_title("airspeed")
    ax[1,0].legend()

    ax[1,1].remove()
    ax[1,1] = fig.add_subplot(2,2,4,projection='3d')
    ax[1,1].plot(lon, lat, alt, label='Aircraft Trajectory')
    ax[1,1].legend()

if __name__ == '__main__':
    # read flight data
    df_temoin = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/flight_data.csv')
    df_stdturb = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/std_turb.csv')

    np_temoin = df_temoin.to_numpy()
    np_stdturb = df_stdturb.to_numpy()

    print(np.array_equal(np_temoin, np_stdturb))

    plot(df_temoin)
    plot(df_stdturb)
    plt.tight_layout()
    plt.show()