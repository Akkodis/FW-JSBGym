import argparse
import random
import torch
import numpy as np
import os
import csv
from tqdm import tqdm

from agents import ppo
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.eval import metrics
from jsbgym.utils import jsbsim_properties as prp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps_no_va.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="ACNoVa-v0", 
        help="the id of the environment")
    parser.add_argument('--train-model', type=str, required=True, 
        help='agent model file name')
    parser.add_argument('--render-mode', type=str, 
        choices=['none','plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/ppo_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--rand-targets', action='store_true', help='set targets randomly')
    parser.add_argument('--severity', type=str, required=True,
                        choices=['off', 'light', 'moderate', 'severe', 'all'],
                        help='severity of the atmosphere (wind and turb)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    np.set_printoptions(suppress=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # seeding
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env setup
    env = ppo.make_env(args.env_id, args.config, args.render_mode, args.tele_file, eval=True)()

    # unwrapped_env = envs.envs[0].unwrapped
    trim_point = TrimPoint('x8')

    train_dict = torch.load(args.train_model, map_location=device)

    # setting the observation normalization parameters
    # env.set_obs_rms(train_dict['norm_obs_rms']['mean'], 
    #                 train_dict['norm_obs_rms']['var'])

    # loading the agent
    ppo_agent = ppo.Agent(env).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()

    # load the reference sequence and initialize the evaluation arrays
    simple_ref_data = np.load("eval/simple_ref_seq_arr.npy")

    # set default target values
    roll_ref: float = simple_ref_data[0, 0]
    pitch_ref: float = simple_ref_data[0, 1]

    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if args.render_mode == "none":
        total_steps = 50_000
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
                       },
                       "gust": {
                           "enable": True
                       }
                   }}

    if args.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [args.severity]

    all_mse = []
    all_rmse = []
    all_fcs_fluct = []

    if not os.path.exists("eval/outputs"):
        os.makedirs("eval/outputs")

    eval_res_csv = f"eval/outputs/eval_res_{os.path.basename(args.train_model)}.csv"
    eval_fieldnames = ["severity", "roll_mse", "pitch_mse", "roll_rmse", 
                        "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct"]

    with open(eval_res_csv, "w") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
        csv_writer.writeheader()

    for i, severity in enumerate(severity_range):
        sim_options["atmosphere"]["severity"] = severity
        e_actions = np.ndarray((total_steps, env.action_space.shape[0]))
        e_obs = np.ndarray((total_steps, env.observation_space.shape[2]))
        eps_fcs_fluct = []
        print(f"********** PPO METRICS {severity} **********")
        obs, _ = env.reset(options=sim_options)
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        ep_cnt = 0 # episode counter
        for step in tqdm(range(total_steps)):
            env.set_target_state(roll_ref, pitch_ref)
            action = ppo_agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
            e_actions[step] = action
            obs, reward, truncated, terminated, info = env.step(action)
            e_obs[step] = info["non_norm_obs"][0, -1] # take the last obs of the history
            obs = torch.Tensor(obs).unsqueeze_(0).to(device)

            done = np.logical_or(truncated, terminated)
            if done:
                ep_cnt += 1
                print(f"Episode reward: {info['episode']['r']}")
                obs, last_info = env.reset()
                obs = torch.Tensor(obs).unsqueeze_(0).to(device)
                ep_fcs_pos_hist = np.array(last_info["fcs_pos_hist"]) # get fcs pos history of the finished episode
                eps_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # get fcs fluctuation of the episode and append it to the list of all fcs fluctuations
                if ep_cnt < len(simple_ref_data):
                    refs = simple_ref_data[ep_cnt]
                roll_ref, pitch_ref = refs[0], refs[1]
        all_fcs_fluct.append(np.mean(np.array(eps_fcs_fluct), axis=0))
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

    # np.save("eval/e_ppo_obs.npy", e_obs)
    # np.save("eval/e_ppo_actions.npy", e_actions)
    env.close()
