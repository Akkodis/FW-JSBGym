import argparse
import random
import gymnasium as gym
import torch
import numpy as np

from agents import ppo
from trim.trim_point import TrimPoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControl-v0", 
        help="the id of the environment")
    parser.add_argument('--train-model', type=str, required=True, 
        help='agent model file name')
    parser.add_argument('--render-mode', type=str, 
        choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/ppo_eval_telemetry.csv", 
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the training params
    train_dict = torch.load(args.train_model, map_location=device)
    seed = train_dict['seed']

    # seeding
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [ppo.make_env(args.env_id, args.config, args.render_mode, args.tele_file, eval=True)]
    )
    unwrapped_env = envs.envs[0].unwrapped
    trim_point = TrimPoint('x8')

    sim_options = {"atmosphere": {"rand_magnitudes": args.rand_atmo_mag, 
                                  "wind": args.wind,
                                  "turb": args.turb}}
    obs, _ = envs.reset(options=sim_options)
    obs = torch.Tensor(obs).to(device)

    # setting the observation normalization parameters
    envs.envs[0].set_obs_rms(train_dict['norm_obs_rms']['mean'], train_dict['norm_obs_rms']['var'])

    # loading the agent
    ppo_agent = ppo.Agent(envs).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()
    episode_reward = 0

    # set default target values
    roll_ref: float = 0.0
    pitch_ref: float = 0.0
    airspeed_ref: float = trim_point.Va_ms

    for step in range(8000):
        if args.rand_targets and step % 500 == 0:
            roll_ref = np.random.uniform(-45, 45) * (np.pi / 180)
            pitch_ref = np.random.uniform(-15, 15) * (np.pi / 180)
            airspeed_ref = np.random.uniform(trim_point.Va_ms - 2, trim_point.Va_ms + 2)
            print("--------------------------------------")
            if args.env_id == "AttitudeControl-v0":
                print(f"roll_ref: {roll_ref}, pitch_ref: {pitch_ref}, airspeed_ref: {airspeed_ref}")
                unwrapped_env.set_target_state(roll_ref, pitch_ref, airspeed_ref)
            if args.env_id == "AttitudeControlNoVa-v0":
                print(f"roll_ref: {roll_ref}, pitch_ref: {pitch_ref}")
                unwrapped_env.set_target_state(roll_ref, pitch_ref)

        action = ppo_agent.get_action_and_value(obs)[1].detach().cpu().numpy()
        obs, reward, truncated, terminated, infos = envs.step(action)
        obs = torch.Tensor(obs).to(device)

        done = np.logical_or(truncated, terminated)
        if done:
            for info in infos["final_info"]:
                print(f"Episode reward: {info['episode']['r']}")

    envs.close()