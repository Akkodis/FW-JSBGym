import argparse
import gymnasium as gym
import numpy as np
import torch
import random
from tqdm import tqdm

from agents.pid import PID
from models import aerodynamics
from trim.trim_point import TrimPoint
from utils.eval_utils import RefSequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eval/ppo_caps_no_va.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="ACNoVa-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['none','plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/pid_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--rand-targets', action='store_true', help='set targets randomly')
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
    kp_roll: float = 1.0
    ki_roll: float = 0.0
    kd_roll: float = 0.5
    roll_pid = PID(
        kp=kp_roll, ki=ki_roll, kd=kd_roll,
        dt=0.01, limit=x8.aileron_limit
    )

    # longitudinal dynamics
    kp_pitch: float = -4.0
    ki_pitch: float = -0.75
    kd_pitch: float = -0.1
    pitch_pid = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=0.01, limit=x8.aileron_limit)

    kp_airspeed: float = 0.5
    ki_airspeed: float = 0.1
    kd_airspeed: float = 0.0
    airspeed_pid = PID(
        kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
        dt=0.01, trim=trim_point,
        limit=x8.throttle_limit, is_throttle=True
    )

    # set default target values
    roll_ref: float = 0.0
    pitch_ref: float = 0.0
    airspeed_ref: float = trim_point.Va_kph

    # load reference sequence and initialize evaluation arrays
    ref_data = np.load("eval/ref_seq_arr.npy")
    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if args.render_mode == "none":
        total_steps = ref_data.shape[0]
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 4000
    sim_options = {"seed": seed,
                   "atmosphere": {
                       "variable": False,
                       "wind": {
                           "enable": True,
                           "rand_continuous": False
                       },
                       "turb": {
                            "enable": True
                       }
                   }}
    severity_options = ["off", "light", "moderate", "severe"]
    pitch_mse_all = np.zeros(len(severity_options))
    roll_mse_all = np.zeros(len(severity_options))

    for i, severity in enumerate(severity_options):
        sim_options["atmosphere"]["severity"] = severity
        e_actions = np.ndarray((total_steps, env.action_space.shape[0]))
        e_obs = np.ndarray((total_steps, env.observation_space.shape[2]))
        print(f"********** PID METRICS {severity} **********")
        obs, _ = env.reset(options=sim_options)
        Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)
        for step in tqdm(range(total_steps)):
            # set random target values
            if args.rand_targets:
                # roll_ref, pitch_ref = refSeq.sample_refs(step)
                refs = ref_data[step]
                roll_ref, pitch_ref = refs[0], refs[1]

            # apply target values
            roll_pid.set_reference(roll_ref)
            pitch_pid.set_reference(pitch_ref)
            env.set_target_state(roll_ref, pitch_ref)

            elevator_cmd, pitch_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
            aileron_cmd, roll_err = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)

            action = np.array([aileron_cmd, elevator_cmd])
            e_actions[step] = action
            obs, reward, truncated, terminated, info = env.step(action)
            e_obs[step] = obs[0, -1]
            Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

            done = np.logical_or(truncated, terminated)
            if done:
                print(f"Episode reward: {info['episode']['r']}")
                obs, _ = env.reset()
                # refSeq.sample_steps(offset=step)
        # compute roll MSE
        roll_mse = np.mean(np.square(e_obs[:, 6]))
        roll_mse_all[i] = roll_mse
        # print(f"Roll MSE: {roll_mse}")
        # compute pitch MSE
        pitch_mse = np.mean(np.square(e_obs[:, 7]))
        pitch_mse_all[i] = pitch_mse
        # print(f"Pitch MSE: {pitch_mse}")

    print("Roll MSEs: ", roll_mse_all)
    print("Pitch MSEs: ", pitch_mse_all)

    np.save("eval/e_pid_obs.npy", e_obs)
    np.save("eval/e_pid_actions.npy", e_actions)