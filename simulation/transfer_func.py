import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import control
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel


mav: AeroModel = AeroModel()

num: np.ndarray = np.array([1, 2])
den: np.ndarray = np.array([1, 0, 4])
H: control.TransferFunction = control.TransferFunction(num, den)

p,z = control.pzmap(H)
print('poles = ', p)
print ('zeros = ', z)
plt.show()