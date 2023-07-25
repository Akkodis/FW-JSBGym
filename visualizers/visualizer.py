import matplotlib.pyplot as plt
import numpy as np
import random
import utils.jsbsim_properties as prp
from typing import Tuple
from utils.jsbsim_properties import BoundedProperty
from simulation.jsb_simulation import Simulation



class PlotVisualizer(object):
    def __init__(self, props_to_print: Tuple[BoundedProperty]) -> None:
        self.props_to_print: Tuple[BoundedProperty] = props_to_print
        self.x = []
        self.y = []
        self.timestep = 0
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.ion()
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        self.fig, self.ax = plt.subplots(3,3)
        self.line1, = self.ax[0,0].plot(self.x, self.y, 'r-')
        # self.line1, = self.ax.plot(self.x, self.y, 'r-')


    def update_plot(self, sim: Simulation):
        self.x.append(self.timestep)
        self.y.append(sim[self.props_to_print[0]])
        self.line1.set_data(self.x, self.y)
        self.line1.axes.relim()
        self.line1.axes.autoscale_view()
        # self.fig.canvas.draw()
        self.ax[0,0].draw_artist(self.line1)
        self.fig.canvas.blit(self.ax[0,0].bbox)
        self.fig.canvas.flush_events()
        self.timestep += 1

