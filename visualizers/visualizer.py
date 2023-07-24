import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from os import path
import random


class PlotVisualizer(object):
    def __init__(self) -> None:
        self.x = []
        self.y = []
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line1, = self.ax.plot(self.x, self.y, 'r-')


    def update_plot(self):
        if len(self.x) == 0:
            self.x.append(0)
            self.y.append(random.randint(0, 256))
        else:
            self.x.append(self.x[-1] + 1)
            self.y.append(random.randint(0, 256))
        self.line1.set_data(self.x, self.y)
        self.line1.axes.relim()
        self.line1.axes.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

