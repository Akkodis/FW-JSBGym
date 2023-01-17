#!/usr/bin/env python3
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd


def animate(i, axis):
    data = pd.read_csv('data/flight_data.csv')
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    axis.cla()
    axis.plot(lon, lat, alt, label='Aircraft Trajectory')
    plt.legend(loc='upper left')
    plt.tight_layout()


ax = plt.figure().add_subplot(projection='3d')
ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, ), interval=100)
plt.show()
