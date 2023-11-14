import argparse
import random
import gymnasium as gym
import torch
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.utils.gym_utils import MyNormalizeObservation
import numpy as np
import ppo_continuous_action as ppo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--train-model', type=str, required=True, help='agent model file name')
    parser.add_argument('--render-mode', type=str, choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
                        help='render mode')
    args = parser.parse_args()
    return args

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, config_file=args.config, render_mode="rgb_array")
        else:
            env = gym.make(env_id, config_file=args.config, render_mode=args.render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = MyNormalizeObservation(env, eval=True)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10)) # TODO : remove ?
        env = gym.wrappers.NormalizeReward(env, gamma=gamma) # TODO : remove ?
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the training params
    train_dict = torch.load(args.train_model, map_location=device)
    seed = train_dict['seed']

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env('AttitudeControlTaskEnv-v0', 1, False, 'ppo_run_1', 0.99)]
    )

    obs, _ = envs.reset(seed=seed)
    obs = torch.Tensor(obs).to(device)
    trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

    # setting the observation normalization parameters
    envs.envs[0].set_obs_rms(train_dict['norm_obs_rms']['mean'], train_dict['norm_obs_rms']['var'])

    # loading the agent
    ppo_agent = ppo.Agent(envs).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()
    episode_reward = 0

    for _ in range(2500):
        # action = env.action_space.sample()
        # action = np.array([[trim_point.elevator, trim_point.aileron, trim_point.throttle]])
        action = ppo_agent.get_action_and_value(obs)[1].detach().cpu().numpy()
        obs, reward, truncated, terminated, infos = envs.step(action)
        obs = torch.Tensor(obs).to(device)

        done = np.logical_or(truncated, terminated)
        if done:
            for info in infos["final_info"]:
                print(f"Episode reward: {info['episode']['r']}")
            break

    envs.close()