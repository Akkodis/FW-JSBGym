import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import jsbgym
from trim.trim_point import TrimPoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControl-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/pid_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--rand-targets', action='store_true', help='set targets randomly')
    parser.add_argument('--rand-atmo-mag', action='store_true', help='randomize the wind and turb magnitudes at each episode')
    parser.add_argument('--turb', action='store_true', help='add turbulence')
    parser.add_argument('--wind', action='store_true', help='add wind')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.env_id == "AttitudeControl-v0":
        args.config = "config/ppo_caps.yaml"
    elif args.env_id == "AttitudeControlNoVa-v0":
        args.config = "config/ppo_caps_no_va.yaml"

    env = gym.make(args.env_id, config_file=args.config, render_mode=args.render_mode, telemetry_file=args.tele_file)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs, _ = env.reset()
    trim_point = TrimPoint('x8')
    throttle_action = []
    elevator_action = []
    aileron_action = []
    throttle_pos = []
    elevator_pos = []
    aileron_pos = []
    action = np.array([0.0, 0.0, 0.0])

    for step in range(500):
        if step == 300:
            # action = np.array([trim_point.aileron, trim_point.elevator, trim_point.throttle])
            action = np.array([0.1, 0.1, 0.1])
        throttle_action.append(action[2])
        elevator_action.append(action[1])
        aileron_action.append(action[0])
        obs, reward, trunc, term, info = env.step(action)
        throttle_pos.append(env.sim["fcs/throttle-pos-norm"])
        elevator_pos.append(env.sim["fcs/elevator-pos-rad"])
        aileron_pos.append(env.sim["fcs/effective-aileron-pos"])

        if term or trunc:
            print("Episode done")
            break

    throttle_action = np.array(throttle_action)
    throttle_pos = np.array(throttle_pos)
    elevator_action = np.array(elevator_action)
    elevator_pos = np.array(elevator_pos)
    aileron_action = np.array(aileron_action)
    aileron_pos = np.array(aileron_pos)

    # plot throttle action and throttle cmd on top of each other
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(throttle_action, label="throttle action")
    ax.plot(throttle_pos, label="throttle pos")
    ax.legend()
    ax.grid()

    ax = fig.add_subplot(132)
    ax.plot(elevator_action, label="elevator action")
    ax.plot(elevator_pos, label="elevator pos")
    ax.legend()
    ax.grid()

    ax = fig.add_subplot(133)
    ax.plot(aileron_action, label="aileron action")
    ax.plot(aileron_pos, label="aileron pos")
    ax.legend()
    ax.grid()

    plt.show()



# if __name__ == '__main__':
#     env_id = "AttitudeControlNoVa-v0" 
#     config = ""
#     if env_id == "AttitudeControl-v0":
#         config = "config/ppo_caps.yaml"
#     elif env_id == "AttitudeControlNoVa-v0":
#         config = "config/ppo_caps_no_va.yaml"

#     env = gym.make(env_id, config, render_mode="plot", telemetry_file="telemetry/sandbox.csv")
#     env = gym.wrappers.RecordEpisodeStatistics(env)

#     obs, _ = env.reset()
#     trim_point = TrimPoint('x8')
#     throttle_action = []
#     throttle_cmd = []
#     action = [0.0, 0.0, 0.0]

#     for step in range(1000):
#         if step == 300:
#             action = [trim_point.aileron, trim_point.elevator, trim_point.throttle]
#         throttle_action.append(action[2])
#         obs, reward, trunc, term, info = env.step(action)
#         throttle_cmd.append(env.sim["fcs/throttle-cmd-norm"])

#         if term or trunc:
#             print("Episode done")
#             break

#     throttle_action = np.array(throttle_action)
#     throttle_cmd = np.array(throttle_cmd)

#     # plot throttle action and throttle cmd on top of each other
#     plt.plot(throttle_action, label="throttle action")
#     plt.plot(throttle_cmd, label="throttle cmd")
#     plt.legend()
#     plt.show()
