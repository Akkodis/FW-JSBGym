import argparse
import gymnasium as gym
import torch
from jsbgym.trim.trim_point import TrimPoint
import numpy as np
import agents.ppo as ppo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--agent-model', type=str, required=True, help='agent model file name')
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
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.expand_dims(obs, axis=0))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # env setup
    env = gym.vector.SyncVectorEnv(
        [make_env('AttitudeControlTaskEnv-v0', 1, False, 'ppo_run_1', 0.99)]
    )

    obs, _ = env.reset(seed=1)
    trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

    ppo_agent = ppo.Agent(envs=env)
    ppo_agent.load_state_dict(torch.load(args.agent_model, map_location=device))
    ppo_agent.eval()
    episode_reward = 0

    for _ in range(2000):
        # action = env.action_space.sample()
        # action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
        action = ppo_agent.get_action_and_value(torch.from_numpy(obs), eval=True)[0].detach().numpy()
        obs, reward, truncated, terminated, info = env.step(action)
        if not(terminated or truncated):
            episode_reward += info['non_norm_reward']

    print(f"Episode reward: {episode_reward}")

    env.close()