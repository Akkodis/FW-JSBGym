#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

data = pd.read_csv('data/flight_data.csv')
lat = data['latitude']
lon = data['longitude']
alt = data['altitude']
roll = data['roll']
pitch = data['pitch']
yaw = data['yaw']

x, y, z = np.array([[0,1],[0,0],[0,0],[0,0]]), np.array([[0,0],[0,1],[0,0],[0,0]]), np.array([[0,0],[0,0],[0,1],[0,0]])
dx, dy, dz = np.array([1,1]), np.array([0,0]), np.array([0,0])

# Create the 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the arrow surface
ax.plot_surface(x, y, z, color='b', rstride=1, cstride=1)
ax.plot_surface(x-dx, y+dy, z+dz, color='b', rstride=1, cstride=1)

plt.show()
