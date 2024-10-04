import sys
from os import path
sys.path.append(f'{path.dirname(path.abspath(__file__))}/..')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from matplotlib.animation import FuncAnimation
from fw_jsbgym.utils import jsbsim_properties as prp


# parse command line arguments
parser = ArgumentParser(description='Plotting Telemetry Data')
parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
parser.add_argument('--tele-file', type=str, required=True, help='Telemetry csv file from where to read and plot data from')
parser.add_argument('--fullscreen', action='store_true', help='True: fullscreen, False: windowed')
args: Namespace = parser.parse_args()

plt.rcParams.update({'font.size': 17})

fig, axis = plt.subplots(2, 2)

df = pd.read_csv(f'{path.dirname(path.abspath(__file__))}/../{args.tele_file}')
nan_arr = np.empty(df.index.size)
nan_arr.fill(np.nan)

lat = df.get(prp.lat_gc_deg.get_legal_name(), default=nan_arr)
lon = df.get(prp.lng_gc_deg.get_legal_name(), default=nan_arr)
alt = df.get(prp.altitude_sl_m.get_legal_name(), default=nan_arr)

roll = df.get(prp.roll_rad.get_legal_name(), default=nan_arr)
pitch = df.get(prp.pitch_rad.get_legal_name(), default=nan_arr)
heading = df.get(prp.heading_rad.get_legal_name(), default=nan_arr)

roll_rate = df.get(prp.p_radps.get_legal_name(), default=nan_arr)
pitch_rate = df.get(prp.q_radps.get_legal_name(), default=nan_arr)
yaw_rate = df.get(prp.r_radps.get_legal_name(), default=nan_arr)

airspeed = df.get(prp.airspeed_kph.get_legal_name(), default=nan_arr)

windspeed_n_kph = df.get(prp.total_windspeed_north_kph.get_legal_name(), default=nan_arr)
windspeed_e_kph = df.get(prp.total_windspeed_east_kph.get_legal_name(), default=nan_arr)
windspeed_d_kph = df.get(prp.total_windspeed_down_kph.get_legal_name(), default=nan_arr)

turb_n_kph = df.get(prp.turb_north_kph.get_legal_name(), default=nan_arr)
turb_e_kph = df.get(prp.turb_east_kph.get_legal_name(), default=nan_arr)
turb_d_kph = df.get(prp.turb_down_kph.get_legal_name(), default=nan_arr)

throttle_cmd = df.get(prp.throttle_cmd.get_legal_name(), default=nan_arr)
elevator_cmd = df.get(prp.elevator_cmd.get_legal_name(), default=nan_arr)
aileron_cmd = df.get(prp.aileron_cmd.get_legal_name(), default=nan_arr)

elevator_pos_norm = df.get(prp.elevator_pos_norm.get_legal_name(), default=nan_arr)
aileron_pos_norm = df.get(prp.aileron_combined_pos_norm.get_legal_name(), default=nan_arr)
throttle_pos = df.get(prp.throttle_pos.get_legal_name(), default=nan_arr)

airspeed_ref = df.get(prp.target_airspeed_kph.get_legal_name(), default=nan_arr)
roll_ref = df.get(prp.target_roll_rad.get_legal_name(), default=nan_arr)
pitch_ref = df.get(prp.target_pitch_rad.get_legal_name(), default=nan_arr)

r_total = df.get(prp.reward_total.get_legal_name(), default=nan_arr)
r_roll = df.get(prp.reward_roll.get_legal_name(), default=nan_arr)
r_pitch = df.get(prp.reward_pitch.get_legal_name(), default=nan_arr)
r_airspeed = df.get(prp.reward_airspeed.get_legal_name(), default=nan_arr)
r_actvar = df.get(prp.reward_actvar.get_legal_name(), default=nan_arr)

kp_roll = df.get(prp.kp_roll.get_legal_name(), default=nan_arr)
ki_roll = df.get(prp.ki_roll.get_legal_name(), default=nan_arr)
kd_roll = df.get(prp.kd_roll.get_legal_name(), default=nan_arr)

kp_pitch = df.get(prp.kp_pitch.get_legal_name(), default=nan_arr)
ki_pitch = df.get(prp.ki_pitch.get_legal_name(), default=nan_arr)
kd_pitch = df.get(prp.kd_pitch.get_legal_name(), default=nan_arr)


num_steps = len(df.index)
tsteps = np.linspace(0, num_steps-1, num=num_steps)


axis[0, 0].plot(tsteps, roll, label='roll' if not np.isnan(np.sum(roll)) else '')
axis[0, 0].plot(tsteps, roll_ref, color='r', linestyle='--', label='roll_ref' if not np.isnan(np.sum(roll_ref)) else '')
axis[0, 0].fill_between(tsteps, roll_ref - np.deg2rad(5), roll_ref + np.deg2rad(5), color='r', alpha=0.2)
axis[0, 0].set_title('roll control')
# axis[0, 0].set_xlabel("timestep")
axis[0, 0].set_ylabel("roll [rad]")
axis[0, 0].legend()
axis[0, 0].grid()

axis[0, 1].plot(tsteps, pitch, label='pitch' if not np.isnan(np.sum(pitch)) else '')
axis[0, 1].plot(tsteps, pitch_ref, color='r', linestyle='--', label='pitch_ref' if not np.isnan(np.sum(pitch_ref)) else '')
axis[0, 1].fill_between(tsteps, pitch_ref - np.deg2rad(5), pitch_ref + np.deg2rad(5), color='r', alpha=0.2)
axis[0, 1].set_title('pitch control')
# axis[0, 1].set_xlabel("timestep")
axis[0, 1].set_ylabel("pitch [rad]")
axis[0, 1].legend()
axis[0, 1].grid()

axis[1, 0].plot(tsteps, aileron_pos_norm, label='aileron_pos_norm' if not np.isnan(np.sum(aileron_pos_norm)) else '')
axis[1, 0].plot(tsteps, elevator_pos_norm, label='elevator_pos_norm' if not np.isnan(np.sum(elevator_pos_norm)) else '')
axis[1, 0].plot(tsteps, throttle_pos, label='throttle_pos' if not np.isnan(np.sum(throttle_pos)) else '')

axis[1, 0].set_title('commands')
axis[1, 0].set_xlabel("timestep")
axis[1, 0].set_ylabel("commands [-]")
axis[1, 0].legend()
axis[1, 0].grid()

# axis[1, 1].plot(tsteps, windspeed_n_kph, label='north' if not np.isnan(np.sum(windspeed_n_kph)) else '')
# axis[1, 1].plot(tsteps, windspeed_e_kph, label='east' if not np.isnan(np.sum(windspeed_e_kph)) else '')
# axis[1, 1].plot(tsteps, windspeed_d_kph, label='down' if not np.isnan(np.sum(windspeed_d_kph)) else '')
# axis[1, 1].set_title('windspeeds')
# axis[1, 1].set_xlabel("timestep")
# axis[1, 1].set_ylabel("windspeed [km/h]")
# axis[1, 1].legend()
# axis[1, 1].grid()


axis[1, 1].plot(tsteps, roll_rate, label='roll_rate' if not np.isnan(np.sum(roll_rate)) else '')
axis[1, 1].plot(tsteps, pitch_rate, label='pitch_rate' if not np.isnan(np.sum(pitch_rate)) else '')
axis[1, 1].plot(tsteps, yaw_rate, label='yaw_rate' if not np.isnan(np.sum(yaw_rate)) else '')
axis[1, 1].set_title('angular velocities')
axis[1, 1].set_xlabel("timestep")
axis[1, 1].set_ylabel("angular velocities [rad/s]")
axis[1, 1].legend()
axis[1, 1].grid()


plt.show()