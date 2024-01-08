# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from time import strftime, localtime
from collections import deque

import jsbgym

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = random.randint(0, 10000)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "uav_rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(5e4) # 50k
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    ts_coef: float = 0.01
    """CAPS: temporal smoothing coefficient"""
    ss_coef: float = 0.1
    """CAPS: spatial smoothing coefficient"""


    # Environment specific arguments
    config: str = "config/ppo_caps_no_va.yaml"
    """the configuration file of the environment"""
    wind: bool = False
    """add wind"""
    turb: bool = False
    """add turbulence"""
    wind_rand_cont: bool = True
    """randomize the wind magnitude continuously"""
    rand_targets: bool = False
    """set targets randomly"""


def make_env(env_id, seed, config, render_mode, telemetry_file=None):
    def thunk():
        env = gym.make(env_id, config_file=config, telemetry_file=telemetry_file,
                           render_mode=render_mode)
        #TODO: add reward and obs normalization ?
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class CNN_extractor(nn.Module): # one separate CNN for each actor and critic
    def __init__(self, env, n_cnn_filters):
        super().__init__()
        self.n_cnn_filters = n_cnn_filters
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.n_cnn_filters,
                      kernel_size=(env.single_observation_space.shape[1], 1), stride=1),
            nn.Tanh(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.cnn_extractor = CNN_extractor(env, n_cnn_filters=3)
        self.fc1 = nn.Linear(env.single_observation_space.shape[2]*self.cnn_extractor.n_cnn_filters + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.cnn_extractor(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.cnn_extractor = CNN_extractor(env, n_cnn_filters=3)
        self.fc1 = nn.Linear(env.single_observation_space.shape[2]*self.cnn_extractor.n_cnn_filters, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.cnn_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )

    args = tyro.cli(Args)
    run_name = f"td3_{args.exp_name}_{args.seed}_{strftime('%d-%m_%H:%M:%S', localtime())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"**** Using Device: {device} ****")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, args.config, "none", None)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    unwr_envs = envs.envs[0].unwrapped

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    sim_options = {"atmosphere": {
                        "variable": True,
                        "wind": {
                            "enable": args.wind,
                            "rand_continuous": args.wind_rand_cont
                        },
                        "turb": {
                            "enable": args.turb
                        }
                   }}
    # TRY NOT TO MODIFY: start the game
    roll_ref = np.random.uniform(np.deg2rad(-30), np.deg2rad(30))
    pitch_ref = np.random.uniform(np.deg2rad(-20), np.deg2rad(20))
    print(f"Initial refs : roll = {roll_ref}, pitch = {pitch_ref}")
    obs, _ = envs.reset(options=sim_options)
    for global_step in range(args.total_timesteps):
        if args.track:
                wandb.log({"global_step": global_step})

        # ALGO LOGIC: put action logic here
        unwr_envs.set_target_state(roll_ref, pitch_ref)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                roll_ref = np.random.uniform(np.deg2rad(-30), np.deg2rad(30))
                pitch_ref = np.random.uniform(np.deg2rad(-20), np.deg2rad(20))
                print(f"Env Done, new ref : roll = {roll_ref}, pitch = {pitch_ref} sampled")
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # CAPS:
                # Temporal Smoothing:
                # act = data.actions
                act = actor(data.observations) # act at time t
                next_act = actor(data.next_observations) # act at time t+1
                # ts_loss = torch.linalg.norm(act - next_act, ord=2)
                ts_loss = F.mse_loss(next_act, act)

                # Spatial Smoothing:
                state_problaw = Normal(data.observations, 0.01)
                state_sampled = state_problaw.sample()
                act_bar = actor(state_sampled)
                # ss_loss = torch.linalg.norm(act - act_bar, ord=2)
                ss_loss = F.mse_loss(act_bar, act)

                td3_actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                total_actor_loss = td3_actor_loss + args.ts_coef * ts_loss + args.ss_coef * ss_loss
                actor_optimizer.zero_grad()
                total_actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/total_actor_loss", total_actor_loss.item(), global_step)
                writer.add_scalar("losses/td3_actor_loss", td3_actor_loss.item(), global_step)
                writer.add_scalar("losses/ts", ts_loss.item(), global_step)
                writer.add_scalar("losses/ts_term", args.ts_coef * ts_loss.item(), global_step)
                writer.add_scalar("losses/ss", ss_loss.item(), global_step)
                writer.add_scalar("losses/ss_term", args.ss_coef * ss_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        save_path: str = "models/train/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = f"{save_path}{run_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.td3_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=(Actor, QNetwork),
        #     device=device,
        #     exploration_noise=args.exploration_noise,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "TD3", f"runs/{run_name}", f"videos/{run_name}-eval")


    print("******** Plotting... ***********")
    actor.eval()
    qf1.eval()
    qf2.eval()

    telemetry_file = f"telemetry/{run_name}.csv"
    obs, _ = envs.reset(options={"render_mode": "log"})
    envs.envs[0].unwrapped.telemetry_setup(telemetry_file)
    roll_ref = np.deg2rad(20)
    pitch_ref = np.deg2rad(15)
    for step in range(4000):
        envs.envs[0].unwrapped.set_target_state(roll_ref, pitch_ref)
        with torch.no_grad():
            e_actions = actor(torch.Tensor(obs).to(device))
            # e_actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            e_actions = e_actions.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        next_obs, _, term, trunc, _ = envs.step(e_actions)
        done = np.logical_or(term, trunc)
        if done:
            break
        obs = next_obs

    telemetry_df = pd.read_csv(telemetry_file)
    telemetry_table = wandb.Table(dataframe=telemetry_df)
    wandb.log({"telemetry": telemetry_table})

    envs.close()
    writer.close()
