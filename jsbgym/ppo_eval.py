import argparse
import random
import gymnasium as gym
import torch
import numpy as np

from agents import ppo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControlTaskEnv-v0", 
        help="the id of the environment")
    parser.add_argument('--train-model', type=str, required=True, 
        help='agent model file name')
    parser.add_argument('--render-mode', type=str, 
        choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument('--turb', action='store_true', help='add turbulence')
    parser.add_argument('--wind', action='store_true', help='add wind')
    args = parser.parse_args()
    return args


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
        [ppo.make_env(args.env_id, args.config, args.render_mode, 0.99, eval=True)]
    )
    unwrapped_env = envs.envs[0].unwrapped

    obs, _ = envs.reset(seed=seed)
    obs = torch.Tensor(obs).to(device)
    if args.turb:
        unwrapped_env.sim['atmosphere/turb-type'] = 3
        unwrapped_env.sim['atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps'] = 75
        unwrapped_env.sim["atmosphere/turbulence/milspec/severity"] = 6

    if args.wind:
        unwrapped_env.sim["atmosphere/wind-north-fps"] = 16.26 * 3.281 # mps to fps
        unwrapped_env.sim["atmosphere/wind-east-fps"] = 16.26 * 3.281 # mps to fps

    # setting the observation normalization parameters
    envs.envs[0].set_obs_rms(train_dict['norm_obs_rms']['mean'], train_dict['norm_obs_rms']['var'])

    # loading the agent
    ppo_agent = ppo.Agent(envs).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()
    episode_reward = 0

    for _ in range(2500):
        action = ppo_agent.get_action_and_value(obs)[1].detach().cpu().numpy()
        obs, reward, truncated, terminated, infos = envs.step(action)
        obs = torch.Tensor(obs).to(device)

        done = np.logical_or(truncated, terminated)
        if done:
            for info in infos["final_info"]:
                print(f"Episode reward: {info['episode']['r']}")
            break

    envs.close()