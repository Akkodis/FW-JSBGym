#!/usr/bin/env python3
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
from os import path


def animate(i, axis) -> None:
    data = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../data/flight_data.csv')
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    axis.cla()
    axis.plot(lon, lat, alt, label='Aircraft Trajectory')
    plt.legend(loc='upper left')
    plt.tight_layout()


ax = plt.figure().add_subplot(projection='3d')
ax.set_aspect('equal')
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, ), interval=50)
plt.show()
