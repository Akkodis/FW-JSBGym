import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import control
import numpy as np
import matplotlib.pyplot as plt
from models.aerodynamics import AeroModel
from trim.trim_point import TrimPoint


trim: TrimPoint = TrimPoint("x8")
uav: AeroModel = AeroModel(trim=trim)

# av1, av2, av3 TF coeffs
av1: float = ((uav.rho * trim.Va_ms * uav.S) / uav.mass) / (uav.CDo + uav.CDalpha * trim.alpha_rad + uav.CDde * trim.elevator)
av2: float = (uav.Pwatt / uav.Khp2w) * uav.Khp2ftlbsec
av3: float = uav.G * np.cos(trim.theta_rad - trim.alpha_rad)

num: list = [[ [av2], [-av3], [1] ]]
den: list = [ [[1, av1], [1, av1], [1, av1]] ]
H: control.TransferFunction = control.tf(num, den)
print(H)
pass