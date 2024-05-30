import random
import torch
import numpy as np
import os
import csv
import hydra

from omegaconf import DictConfig
from agents import sac
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.utils.train_utils import make_env


@hydra.main(version_base=None, config_path="config", config_name="default")
def eval(cfg: DictConfig):
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**** Using Device: {device} ****")

    # seeding
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # shorter cfg aliases
    cfg_sac = cfg.rl.SAC
    cfg_sim = cfg.env.jsbsim
    cfg_task = cfg.env.task

    # env setup
    env = make_env(cfg_sac.env_id, cfg.env, cfg_sim.render_mode,
                       'telemetry/telemetry.csv', eval=True)()

    # unwrapped_env = envs.envs[0].unwrapped
    trim_point = TrimPoint('x8')

    # loading the agent
    train_dict = torch.load(cfg.model_path, map_location=device)[0] # only load the actor's state dict
    sac_agent = sac.Actor_SAC(env).to(device)
    sac_agent.load_state_dict(train_dict)
    sac_agent.eval()

    # load the reference sequence and initialize the evaluation arrays
    simple_ref_data = np.load(f'eval/refs/{cfg.ref_file}.npy')

    # load the jsbsim seeds to apply at each reset and set the first seed
    jsbsim_seeds = np.load(f'eval/refs/jsbsim_seeds.npy')
    cfg_sim.eval_sim_options.seed = float(jsbsim_seeds[0])


    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if cfg_sim.render_mode == "none":
        total_steps = 50_000
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 8000

    if cfg_sim.eval_sim_options.atmosphere.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [cfg_sim.eval_sim_options.atmosphere.severity]

    all_mse = []
    all_rmse = []
    all_fcs_fluct = []

    if not os.path.exists("eval/outputs"):
        os.makedirs("eval/outputs")

    eval_res_csv = f"eval/outputs/{cfg.res_file}.csv"
    eval_fieldnames = ["severity", "roll_mse", "pitch_mse", "roll_rmse", 
                        "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct"]

    with open(eval_res_csv, "w") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
        csv_writer.writeheader()

    for i, severity in enumerate(severity_range):
        cfg_sim.eval_sim_options.atmosphere.severity = severity
        e_obs = []
        eps_fcs_fluct = []
        print(f"********** SAC METRICS {severity} **********")
        obs, _ = env.reset(options=cfg_sim.eval_sim_options)
        ep_cnt = 0 # episode counter
        ep_step = 0 # step counter within an episode
        step = 0
        refs = simple_ref_data[ep_cnt]
        roll_ref, pitch_ref = refs[0], refs[1]
        # set default target values
        # roll_ref: float = np.deg2rad(-15)
        # pitch_ref: float = np.deg2rad(-10)

        while step < total_steps:
            env.set_target_state(roll_ref, pitch_ref)
            action = sac_agent.get_action(torch.Tensor(obs).unsqueeze(0).to(device))[2].squeeze_().detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if cfg_task.mdp.obs_is_matrix:
                e_obs.append(info["non_norm_obs"][0, -1])
            else:
                e_obs.append(info["non_norm_obs"])

            done = np.logical_or(terminated, truncated)
            if done:
                if info['out_of_bounds']:
                    print("Out of bounds")
                    e_obs[len(e_obs)-ep_step:] = [] # delete the last observations if the ep is oob
                    step -= ep_step # set the step counter back to the last episode
                    ep_step = 0 # reset the episode step counter
                else:
                    ep_step = 0 # reset the episode step counter
                    ep_cnt += 1 # increment the episode counter
                print(f"Episode reward: {info['episode']['r']}")
                print(f"******* {step}/{total_steps} *******")
                # break
                obs, last_info = env.reset(options={"seed": float(jsbsim_seeds[ep_cnt])})
                ep_fcs_pos_hist = np.array(last_info["fcs_pos_hist"]) # get fcs pos history of the finished episode
                eps_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # get fcs fluctuation of the episode and append it to the list of all fcs fluctuations
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


if __name__ == '__main__':
    eval()