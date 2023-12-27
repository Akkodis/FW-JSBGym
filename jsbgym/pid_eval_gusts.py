import argparse
from re import A
import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents.pid import PID
from models import aerodynamics
from trim.trim_point import TrimPoint
from utils.eval_utils import RefSequence
from jsbgym.eval import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps_no_va.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="ACNoVa-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['none','plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/pid_gusts_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--save-res-file', action='store_true',default=False, help='save results to file')
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
    if args.env_id == "AC-v0":
        args.config = "config/ppo_caps.yaml"

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
    ki_roll: float = 0.0 # 0.1
    kd_roll: float = 0.5
    roll_pid = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll,
                   dt=0.01, limit=x8.aileron_limit)

    # longitudinal dynamics
    kp_pitch: float = -4.0
    ki_pitch: float = -0.75
    kd_pitch: float = -0.1
    pitch_pid = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=0.01, limit=x8.aileron_limit)

    kp_airspeed: float = 0.5
    ki_airspeed: float = 0.1
    kd_airspeed: float = 0.0
    airspeed_pid: PID = PID(kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
                            dt=0.01, trim=trim_point, limit=x8.throttle_limit, is_throttle=True)

    # total_steps = 50_000
    total_steps = 2000

    atmosphere_setups = {
        "wind OFF, turb OFF, gust OFF": {
            "seed": seed, # moderate constant wind, no turb, no gusts
            "atmosphere": {
                "variable": False,
                "severity": "moderate",
                "wind": {
                    "enable": False,
                    "rand_continuous": False
                },
                "turb": {
                    "enable": False
                },
                "gust": {
                    "enable": False
                }
            }
        },
        # "wind ON, turb OFF, gust OFF": {
        #     "seed": seed, # moderate constant wind, no turb, no gusts
        #     "atmosphere": {
        #         "variable": False,
        #         "severity": "moderate",
        #         "wind": {
        #             "enable": True,
        #             "rand_continuous": False
        #         },
        #         "turb": {
        #             "enable": False
        #         },
        #         "gust": {
        #             "enable": False
        #         }
        #     }
        # },
        # "wind ON, turb ON, gust OFF":{
        #     "seed": seed, # mooderate constant wind, moderate turb, no gusts
        #     "atmosphere": {
        #         "variable": False,
        #         "severity": "moderate",
        #         "wind": {
        #             "enable": True,
        #             "rand_continuous": False
        #         },
        #         "turb": {
        #             "enable": True
        #         },
        #         "gust": {
        #             "enable": False
        #         }
        #     }
        # },
        # "wind ON, turb OFF, gust ON":{
        #     "seed": seed, # mooderate constant wind, moderate turb, no gusts
        #     "atmosphere": {
        #         "variable": False,
        #         "severity": "moderate",
        #         "wind": {
        #             "enable": True,
        #             "rand_continuous": False
        #         },
        #         "turb": {
        #             "enable": False
        #         },
        #         "gust": {
        #             "enable": True
        #         }
        #     }
        # }
        # "wind ON severe, turb ON severe, gust OFF":{
        #     "seed": seed, # mooderate constant wind, moderate turb, no gusts
        #     "atmosphere": {
        #         "variable": False,
        #         "severity": "severe",
        #         "wind": {
        #             "enable": True,
        #             "rand_continuous": False
        #         },
        #         "turb": {
        #             "enable": True
        #         },
        #         "gust": {
        #             "enable": False
        #         }
        #     }
        # },
    }

    pitch_mse_all = np.zeros(len(atmosphere_setups))
    roll_mse_all = np.zeros(len(atmosphere_setups))
    # roll_ref = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
    # pitch_ref = np.random.uniform(np.deg2rad(-30), np.deg2rad(30))
    airspeed_ref = 55
    roll_ref = np.deg2rad(15)
    pitch_ref = np.deg2rad(10)
    pitch_errs = []
    pitch_integerrs = []
    roll_errs = []
    roll_integerrs = []

    for i, (opt_name, sim_options) in enumerate(atmosphere_setups.items()):
        e_actions = np.ndarray((total_steps, env.action_space.shape[0]))
        e_obs = np.ndarray((total_steps, env.observation_space.shape[2]))
        print(f"********** PID METRICS {opt_name} **********")
        obs, _ = env.reset(options=sim_options)
        Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)
        for step in tqdm(range(total_steps)):
            # set random reference values for every episode
            roll_pid.set_reference(roll_ref)
            pitch_pid.set_reference(pitch_ref)
            airspeed_pid.set_reference(airspeed_ref)
            env.set_target_state(roll_ref, pitch_ref, airspeed_ref)

            elevator_cmd, pitch_err, pitch_integerr = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
            aileron_cmd, roll_err, roll_integerr = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)
            throttle_cmd, airspeed_err, _ = airspeed_pid.update(state=Va, saturate=True)
            pitch_errs.append(pitch_err)
            pitch_integerrs.append(pitch_integerr)
            roll_errs.append(roll_err)
            roll_integerrs.append(roll_integerr)

            action = np.array([aileron_cmd, elevator_cmd, throttle_cmd])
            e_actions[step] = action
            obs, reward, truncated, terminated, info = env.step(action)
            e_obs[step] = obs[0, -1]
            Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

            # if step == 500 + total_steps:
            #     env.gust_start()

            done = np.logical_or(truncated, terminated)
            if done:
                print(f"Episode reward: {info['episode']['r']}")
                # roll_ref = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
                # pitch_ref = np.random.uniform(np.deg2rad(-30), np.deg2rad(30))
                # roll_ref = np.deg2rad(55)
                # pitch_ref = np.deg2rad(20)
                obs, _ = env.reset()
                # refSeq.sample_steps(offset=step)
        roll_mse_all[i] = np.mean(np.square(e_obs[:, 6]))
        pitch_mse_all[i] = np.mean(np.square(e_obs[:, 7]))

    for atmo_setup, roll_mse, pitch_mse in zip(atmosphere_setups.keys(), roll_mse_all, pitch_mse_all):
        print(f"{atmo_setup} - roll MSE: {roll_mse}, pitch MSE: {pitch_mse}")
    

    # plot errors and integral errors
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(pitch_errs)
    axs[0, 0].set_title('Pitch Errors')
    axs[0, 1].plot(pitch_integerrs)
    axs[0, 1].set_title('Pitch Integral Errors')
    axs[1, 0].plot(roll_errs)
    axs[1, 0].set_title('Roll Errors')
    axs[1, 1].plot(roll_integerrs)
    axs[1, 1].set_title('Roll Integral Errors')
    fig.tight_layout()
    plt.show()

    # np.save("eval/e_pid_obs.npy", e_obs)
    # np.save("eval/e_pid_actions.npy", e_actions)