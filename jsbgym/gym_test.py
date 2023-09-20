import gymnasium as gym
import torch
from jsbgym.trim.trim_point import TrimPoint
import numpy as np
import agents.ppo as ppo

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, config_file='config/bohn_ppo.yaml', render_mode="rgb_array")
        else:
            env = gym.make(env_id, config_file='config/bohn_ppo.yaml', render_mode="fgear_plot")
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# env setup
env = gym.vector.SyncVectorEnv(
    [make_env('AttitudeControlTaskEnv-v0', 1, False, 'ppo_run_1', 0.99)]
)

obs, _ = env.reset(seed=1)
trim_point: TrimPoint = TrimPoint(aircraft_id='x8')

ppo_agent = ppo.Agent(envs=env)
ppo_agent.load_state_dict(torch.load('models/AttitudeControlTaskEnv-v0__ppo_continuous_action__1__1695119387.pt', map_location=device))
ppo_agent.eval()

for _ in range(1200):
    # action = env.action_space.sample()
    # action = np.array([trim_point.elevator, trim_point.aileron, trim_point.throttle])
    action = ppo_agent.get_action_and_value(torch.from_numpy(obs))[0].detach().numpy()
    obs, reward, truncated, terminated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset(seed=1)

env.close()