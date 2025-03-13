#!/usr/bin/env python3
import sys
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from matplotlib.animation import FuncAnimation
from fw_jsbgym.utils import jsbsim_properties as prp



def animate(i, axis, tele_file) -> None:
    df = pd.read_csv(tele_file)
    nan_arr = np.empty(df.index.size)
    nan_arr.fill(np.nan)

    lat = df.get(prp.lat_gc_deg.get_legal_name(), default=nan_arr)
    lon = df.get(prp.lng_gc_deg.get_legal_name(), default=nan_arr)
    alt = df.get(prp.altitude_sl_m.get_legal_name(), default=nan_arr)

    enu_x_m = df.get(prp.enu_x_m.get_legal_name(), default=nan_arr)
    enu_y_m = df.get(prp.enu_y_m.get_legal_name(), default=nan_arr)
    enu_z_m = df.get(prp.enu_z_m.get_legal_name(), default=nan_arr)

    ecef_x_m = df.get(prp.ecef_x_m.get_legal_name(), default=nan_arr)
    ecef_y_m = df.get(prp.ecef_y_m.get_legal_name(), default=nan_arr)
    ecef_z_m = df.get(prp.ecef_z_m.get_legal_name(), default=nan_arr)

    target_enu_x_m = df.get(prp.target_enu_x_m.get_legal_name(), default=nan_arr)
    target_enu_y_m = df.get(prp.target_enu_y_m.get_legal_name(), default=nan_arr)
    target_enu_z_m = df.get(prp.target_enu_z_m.get_legal_name(), default=nan_arr)

    roll = df.get(prp.roll_rad.get_legal_name(), default=nan_arr)
    pitch = df.get(prp.pitch_rad.get_legal_name(), default=nan_arr)
    heading = df.get(prp.heading_rad.get_legal_name(), default=nan_arr)

    roll_rate = df.get(prp.p_radps.get_legal_name(), default=nan_arr)
    pitch_rate = df.get(prp.q_radps.get_legal_name(), default=nan_arr)
    yaw_rate = df.get(prp.r_radps.get_legal_name(), default=nan_arr)

    airspeed = df.get(prp.airspeed_kph.get_legal_name(), default=nan_arr)

    dist_to_target = df.get(prp.dist_to_target_m.get_legal_name(), default=nan_arr)

    windspeed_n_kph = df.get(prp.total_windspeed_north_kph.get_legal_name(), default=nan_arr)
    windspeed_e_kph = df.get(prp.total_windspeed_east_kph.get_legal_name(), default=nan_arr)
    windspeed_d_kph = df.get(prp.total_windspeed_down_kph.get_legal_name(), default=nan_arr)

    turb_n_kph = df.get(prp.turb_north_kph.get_legal_name(), default=nan_arr)
    turb_e_kph = df.get(prp.turb_east_kph.get_legal_name(), default=nan_arr)
    turb_d_kph = df.get(prp.turb_down_kph.get_legal_name(), default=nan_arr)

    throttle_cmd = df.get(prp.throttle_cmd.get_legal_name(), default=nan_arr)
    elevator_cmd = df.get(prp.elevator_cmd.get_legal_name(), default=nan_arr)
    aileron_cmd = df.get(prp.aileron_cmd.get_legal_name(), default=nan_arr)

    elevator_pos_rad = df.get(prp.elevator_pos_rad.get_legal_name(), default=nan_arr)
    aileron_pos_rad = df.get(prp.aileron_combined_pos_rad.get_legal_name(), default=nan_arr)
    elevator_pos_norm = df.get(prp.elevator_pos_norm.get_legal_name(), default=nan_arr)
    aileron_pos_norm = df.get(prp.aileron_combined_pos_norm.get_legal_name(), default=nan_arr)
    throttle_pos = df.get(prp.throttle_pos.get_legal_name(), default=nan_arr)

    u_kph = df.get(prp.u_kph.get_legal_name(), default=nan_arr)
    v_kph = df.get(prp.v_kph.get_legal_name(), default=nan_arr)
    w_kph = df.get(prp.w_kph.get_legal_name(), default=nan_arr)

    airspeed_ref = df.get(prp.target_airspeed_kph.get_legal_name(), default=nan_arr)

    alpha = df.get(prp.alpha_rad.get_legal_name(), default=nan_arr)
    beta = df.get(prp.beta_rad.get_legal_name(), default=nan_arr)

    r_total = df.get(prp.reward_total.get_legal_name(), default=nan_arr)
    r_dist = df.get(prp.reward_dist.get_legal_name(), default=nan_arr)
    r_airspeed = df.get(prp.reward_airspeed.get_legal_name(), default=nan_arr)
    r_actvar = df.get(prp.reward_actvar.get_legal_name(), default=nan_arr)


    for(dim_1) in axis:
        for(dim_2) in dim_1:
            dim_2.cla()

    num_steps = len(df.index)
    tsteps = np.linspace(0, num_steps-1, num=num_steps)

    if df.index.size > 0:
        axis[0, 0].plot(tsteps, alt, label='altitude' if not np.isnan(np.sum(alt)) else '')
        # axis[0, 0].set_xlabel("timestep")
        axis[0, 0].set_ylabel("altitude [m]")
        axis[0, 0].set_title("altitude control")
        axis[0, 0].legend()
        axis[0, 0].grid()

        if df.index.size > 1:
            axis[0, 1].plot(tsteps, roll, label='roll' if not np.isnan(np.sum(roll)) else '')
            axis[0, 1].plot(tsteps, pitch, label='pitch' if not np.isnan(np.sum(pitch)) else '')
            axis[0, 1].plot(tsteps, heading, label='heading' if not np.isnan(np.sum(heading)) else '')
            axis[0, 1].set_title('Euler Angles')
            # axis[0, 1].set_xlabel("timestep")
            axis[0, 1].set_ylabel("[rad]")
            axis[0, 1].legend()
            axis[0, 1].grid()

        # axis[0, 1].set_title("Heading (psi) control [rad]")
        # # axis[0, 1].set_xlabel("timestep")
        # axis[0, 1].set_ylabel("heading [rad]")
        # axis[0, 1].legend()
        # axis[0, 1].grid()

        axis[0, 2].plot(tsteps, airspeed, label='airspeed' if not np.isnan(np.sum(airspeed)) else '')
        axis[0, 2].plot(tsteps, airspeed_ref, color='r', linestyle='--', label='airspeed_ref' if not np.isnan(np.sum(airspeed_ref)) else '')
        axis[0, 2].set_title('airspeed control')
        # axis[0, 2].set_xlabel("timestep")
        axis[0, 2].set_ylabel("airspeed [km/h]")
        axis[0, 2].legend()
        axis[0, 2].grid()

        if df.index.size > 2:
            axis[1, 0].plot(enu_x_m[0], enu_y_m[0], enu_z_m[0], 'ro', label='start')
            axis[1, 0].plot(target_enu_x_m[1], target_enu_y_m[1], target_enu_z_m[1], 'go', label='target')
            axis[1, 0].plot(enu_x_m, enu_y_m, enu_z_m)
            axis[1, 0].set_title('ENU Trajectory [m]')
            axis[1, 0].set_xlabel("x")
            axis[1, 0].set_ylabel("y")
            axis[1, 0].set_zlabel("z")
            axis[1, 0].legend()

        # wait for the telemetry file to be filled with some data so that the plotter doesn't crash when computing scale bounds
        x_min, x_max = min(enu_x_m.min(), target_enu_x_m.min()) - 10, max(enu_x_m.max(), target_enu_x_m.max()) + 10
        y_min, y_max = min(enu_y_m.min(), target_enu_y_m.min()) - 10, max(enu_y_m.max(), target_enu_y_m.max()) + 10
        z_min, z_max = min(enu_z_m.min(), target_enu_z_m.min()) - 10, max(enu_z_m.max(), target_enu_z_m.max()) + 10
        axis[1, 0].set_xlim(x_min, x_max)
        axis[1, 0].set_ylim(y_min, y_max)
        axis[1, 0].set_zlim(z_min, z_max)

        axis[1, 1].plot(tsteps, dist_to_target, label='distance to target' if not np.isnan(np.sum(dist_to_target)) else '')
        axis[1, 1].set_title('distance to target')
        axis[1, 1].set_ylabel('distance [m]')
        axis[1, 1].legend()
        axis[1, 1].grid()

        axis[1, 2].plot(tsteps, r_dist, label='r_distance' if not np.isnan(np.sum(r_dist)) else '')
        axis[1, 2].plot(tsteps, r_airspeed, label='r_airspeed' if not np.isnan(np.sum(r_airspeed)) else '')
        axis[1, 2].plot(tsteps, r_actvar, label='r_actvar' if not np.isnan(np.sum(r_actvar)) else '')
        axis[1, 2].plot(tsteps, r_total, label='r_total' if not np.isnan(np.sum(r_total)) else '')
        axis[1, 2].set_title('rewards')
        axis[1, 2].set_ylabel('reward [-]')
        axis[1, 2].legend()
        axis[1, 2].grid()

        axis[2, 0].plot(tsteps, u_kph, label='u' if not np.isnan(np.sum(u_kph)) else '')
        axis[2, 0].plot(tsteps, v_kph, label='v' if not np.isnan(np.sum(v_kph)) else '')
        axis[2, 0].plot(tsteps, w_kph, label='w' if not np.isnan(np.sum(w_kph)) else '')
        axis[2, 0].set_title('body velocities')
        axis[2, 0].set_xlabel("timestep")
        axis[2, 0].set_ylabel("body velocities [km/h]")
        axis[2, 0].legend()
        axis[2, 0].grid()

        # axis[2, 0].plot(tsteps, aileron_cmd, label='aileron_cmd' if not np.isnan(np.sum(aileron_cmd)) else '')
        # axis[2, 0].plot(tsteps, elevator_cmd, label='elevator_cmd' if not np.isnan(np.sum(elevator_cmd)) else '')
        # axis[2, 0].plot(tsteps, throttle_cmd, label='throttle_cmd' if not np.isnan(np.sum(throttle_cmd)) else '')
        # # axis[2, 0].plot(tsteps, aileron_pos_norm, label='aileron_pos_norm' if not np.isnan(np.sum(aileron_pos_norm)) else '')
        # # axis[2, 0].plot(tsteps, elevator_pos_norm, label='elevator_pos_norm' if not np.isnan(np.sum(elevator_pos_norm)) else '')
        # # axis[2, 0].plot(tsteps, throttle_pos, label='throttle_pos' if not np.isnan(np.sum(throttle_pos)) else '')
        # axis[2, 0].set_title('commands')
        # axis[2, 0].set_xlabel("timestep")
        # axis[2, 0].set_ylabel("commands [-]")
        # axis[2, 0].legend()
        # axis[2, 0].grid()

        axis[2, 1].plot(tsteps, roll_rate, label='roll_rate' if not np.isnan(np.sum(roll_rate)) else '')
        axis[2, 1].plot(tsteps, pitch_rate, label='pitch_rate' if not np.isnan(np.sum(pitch_rate)) else '')
        axis[2, 1].plot(tsteps, yaw_rate, label='yaw_rate' if not np.isnan(np.sum(yaw_rate)) else '')
        axis[2, 1].set_title('angular velocities')
        axis[2, 1].set_xlabel("timestep")
        axis[2, 1].set_ylabel("angular velocities [rad/s]")
        axis[2, 1].legend()
        axis[2, 1].grid()

        axis[2, 2].plot(tsteps, windspeed_n_kph, label='north' if not np.isnan(np.sum(windspeed_n_kph)) else '')
        axis[2, 2].plot(tsteps, windspeed_e_kph, label='east' if not np.isnan(np.sum(windspeed_e_kph)) else '')
        axis[2, 2].plot(tsteps, windspeed_d_kph, label='down' if not np.isnan(np.sum(windspeed_d_kph)) else '')
        axis[2, 2].set_title('windspeeds')
        axis[2, 2].set_xlabel("timestep")
        axis[2, 2].set_ylabel("windspeed [km/h]")
        axis[2, 2].legend()
        axis[2, 2].grid()

        # axis[2, 2].plot(tsteps, alpha, label='alpha' if not np.isnan(np.sum(alpha)) else '')
        # axis[2, 2].plot(tsteps, alpha_dec, label='alpha_dec' if not np.isnan(np.sum(alpha_dec)) else '')
        # axis[2, 2].plot(tsteps, beta, label='beta' if not np.isnan(np.sum(beta)) else '')
        # axis[2, 2].plot(tsteps, beta_dec, label='beta_dec' if not np.isnan(np.sum(beta_dec)) else '')
        # axis[2, 2].plot(tsteps, alpha_im_dec, label='alpha_im_dec' if not np.isnan(np.sum(alpha_im_dec)) else '')
        # axis[2, 2].plot(tsteps, beta_im_dec, label='beta_im_dec' if not np.isnan(np.sum(beta_im_dec)) else '')
        # axis[2, 2].set_xlabel("timestep")
        # axis[2, 2].set_title('AoA and Sideslip')
        # axis[2, 2].legend()
        # axis[2, 2].grid()


def setup_axes():
    # setting up axis for animation
    fig, ax = plt.subplots(3, 3)
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Setting 3D subplot for trajectory plot
    ax[1, 0].remove()
    ax[1, 0] = fig.add_subplot(3, 3, 4, projection='3d')
    return ax


def main():
    # parse command line arguments
    parser = ArgumentParser(description='Plotting Telemetry Data')
    parser.add_argument('--scale', action='store_true', help='True: keep aspect ratio, False: scale to fit data (for trajectory plot)')
    parser.add_argument('--tele-file', type=str, required=True, help='Telemetry csv file from where to read and plot data from')
    parser.add_argument('--animate', action='store_true', help='True: animate, False: static plot at the end of the simulation')
    args: Namespace = parser.parse_args()

    ax = setup_axes()

    # starting animation
    if args.animate:
        ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, args.tele_file, ), interval=50, blit=False)
        print("TELEMETRY FILE : ", args.tele_file, file=sys.stderr)
        print("Animation plot started...", file=sys.stderr)
        plt.show()
    elif not args.animate:
        animate(0, ax, args.tele_file)
        plt.show()


if __name__ == '__main__':
    main()
