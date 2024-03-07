import argparse
import gymnasium as gym
import numpy as np
import torch
import random
import os
import csv

from agents.pid import PID
from models import aerodynamics
from trim.trim_point import TrimPoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps_no_va.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="ACNoVa-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['none','plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/pid_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--ref-file', type=str, required=True,
                        help='reference sequence file')
    parser.add_argument('--severity', type=str, required=True,
                        choices=['off', 'light', 'moderate', 'severe', 'all'],
                        help='severity of the atmosphere (wind and turb)')
    parser.add_argument('--out-file', type=str, default='eval_res_pid.csv', 
                        help='save results to file')
    parser.add_argument('--rand-fdm', action='store_true',
                        help='randomize the fdm coefs at the start of each episode')
    args = parser.parse_args()
    return args


def rearrange_obs(obs: np.ndarray) -> tuple[float, float, float, float, float]:
    roll = obs[0][4][0]
    pitch = obs[0][4][1]
    Va = obs[0][4][2]
    roll_rate = obs[0][4][3]
    pitch_rate = obs[0][4][4]
    return Va, roll, pitch, roll_rate, pitch_rate


if __name__ == '__main__':
    args = parse_args()
    np.set_printoptions(precision=3)

    # seeding
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    env = gym.make(args.env_id, config_file=args.config, render_mode=args.render_mode,
                    telemetry_file=args.tele_file)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # refSeq = RefSequence(num_refs=5)
    # refSeq.sample_steps()

    x8 = aerodynamics.AeroModel()
    trim_point = TrimPoint('x8')

    # setting handtunes PID gains
    # lateral dynamics
    kp_roll: float = 1.5
    ki_roll: float = 0.1
    kd_roll: float = 0.1

    # kp_roll: float = 1.0
    # ki_roll: float = 0.0
    # kd_roll: float = 0.5
    roll_pid = PID(
        kp=kp_roll, ki=ki_roll, kd=kd_roll,
        dt=0.01, limit=x8.aileron_limit
    )

    # longitudinal dynamics
    kp_pitch: float = -2.0
    ki_pitch: float = -0.3
    kd_pitch: float = -0.1

    # kp_pitch: float = -4.0
    # ki_pitch: float = -0.75
    # kd_pitch: float = -0.1

    pitch_pid = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=0.01, limit=x8.elevator_limit
                    )

    # load reference sequence and initialize evaluation arrays
    simple_ref_data = np.load(args.ref_file)

    # set default target values
    roll_ref: float = simple_ref_data[0, 0]
    pitch_ref: float = simple_ref_data[0, 1]

    # roll_ref: float = np.deg2rad(58)
    # pitch_ref: float = np.deg2rad(28)

    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if args.render_mode == "none":
        total_steps = 50_000
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 8000
    sim_options = {"seed": seed,
                   "atmosphere": {
                       "variable": False,
                       "wind": {
                           "enable": True,
                           "rand_continuous": False
                       },
                       "turb": {
                            "enable": True
                       },
                       "gust": {
                            "enable": True
                       },
                    },
                   "rand_fdm": {
                       "enable": args.rand_fdm,
                   }
                  }

    if args.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [args.severity]

    all_mse = []
    all_rmse = []
    all_fcs_fluct = []

    if not os.path.exists("eval/outputs"):
        os.makedirs("eval/outputs")

    eval_res_csv = f"eval/outputs/{args.out_file}.csv"
    eval_fieldnames = ["severity", "roll_mse", "pitch_mse", "roll_rmse", 
                        "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct"]

    with open(eval_res_csv, "w") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
        csv_writer.writeheader()

    print(f"min roll: {np.min(simple_ref_data[:, 0])}, max roll: {np.max(simple_ref_data[:, 0])}")
    print(f"min pitch: {np.min(simple_ref_data[:, 1])}, max pitch: {np.max(simple_ref_data[:, 1])}")

    for i, severity in enumerate(severity_range):
        sim_options["atmosphere"]["severity"] = severity
        e_obs = []
        eps_fcs_fluct = []
        print(f"********** PID METRICS {severity} **********")
        obs, _ = env.reset(options=sim_options)
        Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)
        ep_cnt = 0 # episode counter
        ep_step = 0
        step = 0
        refs = simple_ref_data[ep_cnt]
        roll_ref, pitch_ref = refs[0], refs[1]
        while step < total_steps:
            # apply target values
            roll_pid.set_reference(roll_ref)
            pitch_pid.set_reference(pitch_ref)
            env.set_target_state(roll_ref, pitch_ref)

            aileron_cmd, roll_err, _ = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)
            elevator_cmd, pitch_err, _ = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)

            action = np.array([aileron_cmd, elevator_cmd])
            obs, reward, truncated, terminated, info = env.step(action)
            e_obs.append(obs[0, -1])
            Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

            done = np.logical_or(truncated, terminated)
            if done:
                if info['out_of_bounds']:
                    print("Out of bounds")
                    e_obs[len(e_obs)-ep_step:] = [] # delete last ep obs if out of bounds
                    step -= ep_step
                    ep_step = 0
                else:
                    ep_step = 0
                    ep_cnt += 1
                print(f"Episode reward: {info['episode']['r']}")
                print(f"******* {step}/{total_steps} *******")
                # break
                obs, last_info = env.reset()
                ep_fcs_pos_hist = np.array(last_info["fcs_pos_hist"]) # get fcs pos history of the finished episode
                eps_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # get fcs fluctuation of the episode and append it to the list of all fcs fluctuations
                pitch_pid.reset()
                roll_pid.reset()
                if ep_cnt < len(simple_ref_data):
                    refs = simple_ref_data[ep_cnt]
                roll_ref, pitch_ref = refs[0], refs[1]
            ep_step += 1
            step += 1

        all_fcs_fluct.append(np.mean(np.array(eps_fcs_fluct), axis=0))
        e_obs = np.array(e_obs)
        print(f"e_obs shape: {e_obs.shape}")
        print(f"eps_fcs_fluct shape: {np.array(eps_fcs_fluct).shape}")
        roll_mse = np.mean(np.square(e_obs[:, 6]))
        pitch_mse = np.mean(np.square(e_obs[:, 7]))
        all_mse.append([roll_mse, pitch_mse])
        roll_rmse = np.sqrt(roll_mse)
        pitch_rmse = np.sqrt(pitch_mse)
        all_rmse.append([roll_rmse, pitch_rmse])

    for mse, rmse, fcs_fluct, severity in zip(all_mse, all_rmse, all_fcs_fluct, severity_range):
        print("\nSeverity: ", severity)
        print(f"  Roll MSE: {mse[0]:.4f}\n  Pitch MSE: {mse[1]:.4f}")
        print(f"  Roll RMSE: {rmse[0]:.4f}\n  Pitch RMSE: {rmse[1]:.4f}")
        print(f"  Roll fluctuation: {fcs_fluct[0]:.4f}\n  Pitch fluctuation: {fcs_fluct[1]:.4f}")
        with open(eval_res_csv, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
            csv_writer.writerow({"severity": severity, "roll_mse": mse[0], "pitch_mse": mse[1], 
                                "roll_rmse": rmse[0], "pitch_rmse": rmse[1], 
                                "roll_fcs_fluct": fcs_fluct[0], "pitch_fcs_fluct": fcs_fluct[1]})

    env.close()