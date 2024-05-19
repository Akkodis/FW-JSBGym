import os
import numpy as np
import torch
import gymnasium as gym
from jsbgym.agents.sac import Actor_SAC
from jsbgym.agents.ppo import Agent_PPO

# Global variables
# Sequence of roll and pitch references for the the periodic evaluation
ref_seq: np.ndarray = np.array([
												[	# roll			,pitch
													[np.deg2rad(25), np.deg2rad(15)], # easy
													[np.deg2rad(-25), np.deg2rad(-15)],
													[np.deg2rad(25), np.deg2rad(-15)],
													[np.deg2rad(-25), np.deg2rad(15)]
												],
												[
													[np.deg2rad(40), np.deg2rad(22)], # medium
													[np.deg2rad(-40), np.deg2rad(-22)],
													[np.deg2rad(40), np.deg2rad(-22)],
													[np.deg2rad(-40), np.deg2rad(22)]
												],
												[
													[np.deg2rad(55), np.deg2rad(28)], # hard
													[np.deg2rad(-55), np.deg2rad(-28)],
													[np.deg2rad(55), np.deg2rad(-28)],
													[np.deg2rad(-55), np.deg2rad(28)]
												]
											])

# Run periodic evaluation during training
def periodic_eval(cfg_mdp, env, agent, device):
    """Evaluate a TD-MPC2 agent."""
    print("*** Evaluating the agent ***")
    env.eval = True
    ep_rewards = []
    dif_obs = []
    dif_fcs_fluct = [] # dicts storing all obs across all episodes and fluctuation of the flight controls for all episodes
    for dif_idx, ref_dif in enumerate(ref_seq): # iterate over the difficulty levels
        dif_obs.append([])
        dif_fcs_fluct.append([])
        for ref_idx, ref_ep in enumerate(ref_dif): # iterate over the ref for 1 episode
            obs, info = env.reset()
            obs, info, done, ep_reward = torch.Tensor(obs).unsqueeze(0).to(device), info, False, 0
            while not done:
                # Set roll and pitch references
                env.set_target_state(ref_ep[0], ref_ep[1]) # 0: roll, 1: pitch
                with torch.no_grad():
                    if isinstance(agent, Actor_SAC):
                        action = agent.get_action(obs)[2].squeeze_(0).detach().cpu().numpy()
                    elif isinstance(agent, Agent_PPO):
                        action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
                obs, reward, term, trunc, info = env.step(action)
                obs = torch.Tensor(obs).unsqueeze(0).to(device)
                done = np.logical_or(term, trunc)
                dif_obs[dif_idx].append(info['non_norm_obs']) # append the non-normalized observation to the list
                ep_reward += info['non_norm_reward']

            ep_fcs_pos_hist = np.array(info['fcs_pos_hist'])
            dif_fcs_fluct[dif_idx].append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # compute the fcs fluctuation of the episode being reset and append to the list

            ep_rewards.append(ep_reward)
    env.reset()

    # computing the mean fcs fluctuation across all episodes for each difficulty level
    dif_fcs_fluct = np.array(dif_fcs_fluct)
    easy_fcs_fluct = np.mean(np.array(dif_fcs_fluct[0]), axis=0)
    medium_fcs_fluct = np.mean(np.array(dif_fcs_fluct[1]), axis=0)
    hard_fcs_fluct = np.mean(np.array(dif_fcs_fluct[2]), axis=0)

    # computing the rmse of the roll and pitch angles across all episodes for each difficulty level
    obs_hist_size = cfg_mdp.obs_hist_size
    dif_obs = np.array(dif_obs)
    if obs_hist_size == 1 and not cfg_mdp.obs_is_matrix:
        easy_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 6])))
        easy_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 7])))
        medium_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 6])))
        medium_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 7])))
        hard_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 6])))
        hard_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 7])))
    elif obs_hist_size > 1 and cfg_mdp.obs_is_matrix:
        easy_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, :, obs_hist_size-1, 6])))
        easy_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, :, obs_hist_size-1, 7])))
        medium_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, :, obs_hist_size-1, 6])))
        medium_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, :, obs_hist_size-1, 7])))
        hard_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, :, obs_hist_size-1, 6])))
        hard_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, :, obs_hist_size-1, 7])))
    env.eval = False

    return dict(
        episode_reward=np.nanmean(ep_rewards),
        easy_roll_rmse=easy_roll_rmse,
        easy_pitch_rmse=easy_pitch_rmse,
        medium_roll_rmse=medium_roll_rmse,
        medium_pitch_rmse=medium_pitch_rmse,
        hard_roll_rmse=hard_roll_rmse,
        hard_pitch_rmse=hard_pitch_rmse,
        easy_ail_fluct=easy_fcs_fluct[0],
        easy_ele_fluct=easy_fcs_fluct[1],
        medium_ail_fluct=medium_fcs_fluct[0],
        medium_ele_fluct=medium_fcs_fluct[1],
        hard_ail_fluct=hard_fcs_fluct[0],
        hard_ele_fluct=hard_fcs_fluct[1],
    )


def make_env(env_id, cfg_env, render_mode, telemetry_file=None, eval=False, gamma=0.99, run_name='', idx=0):
    def thunk():
        env = gym.make(env_id, cfg_env=cfg_env, telemetry_file=telemetry_file,
                        render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = MyNormalizeObservation(env, eval=eval)
        if not eval:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk


# Save the model PPO
def save_model_PPO(save_path, run_name, agent, env, seed):
    save_path: str = "models/train/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = f"{save_path}{run_name}.pt"
    train_dict = {}
    train_dict["seed"] = seed
    train_dict["agent"] = agent.state_dict()
    torch.save(train_dict, f"{save_path}{run_name}.pt")
    print(f"agent saved to {model_path}")


# Save the model TD3/SAC
def save_model_SAC(run_name, actor, qf1, qf2):
    save_path: str = "models/train/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = f"{save_path}{run_name}.pt"
    torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
    print(f"agent saved to {model_path}")
