# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from time import strftime, localtime
from distutils.util import strtobool
from tqdm import tqdm
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.agents import ppo
from jsbgym.utils.eval_utils import RefSequence

import wandb
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=random.randint(0, 100000),
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo_uav",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--no-eval", action='store_true', default=False, help="do not evaluate the agent at the end of training")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ACNoVa-v0",
        help="the id of the environment")
    parser.add_argument("--config", type=str, default="config/ppo_caps_no_va.yaml",
        help="the config file of the environnement")
    parser.add_argument("--total-timesteps", type=float, default=1.5e6,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--ts-coef", type=float, default=5e-2,
        help="coefficient of the temporal smoothing loss function")
    parser.add_argument("--ss-coef", type=float, default=1e-1,
        help="coefficient of the spatial smoothing loss function")
    parser.add_argument("--pa-coef", type=float, default=1e-4,
        help="coefficient of the spatial smoothing loss function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument('--save-best',type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                         help='save only the best models')

    # training sim specific arguments
    parser.add_argument('--rand-targets',type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help='set targets randomly')
    parser.add_argument('--wind-rand-cont',type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help='randomize the wind magnitude continuously')
    parser.add_argument('--turb', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help='add turbulence')
    parser.add_argument('--wind', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help='add wind')
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

# TODO: Implement save_model function
def save_model(save_path, run_name, agent, env, seed):
    print("saving agent...")
    train_dict = {}
    train_dict["norm_obs_rms"] = {"mean": env.get_obs_rms().mean, "var": env.get_obs_rms().var}
    train_dict["seed"] = seed
    train_dict["agent"] = agent.state_dict()
    torch.save(train_dict, f"{save_path}{run_name}.pt")

if __name__ == "__main__":
    args = parse_args()
    args.total_timesteps = int(args.total_timesteps)
    np.set_printoptions(suppress=True)

    if args.env_id == "AttitudeControl-v0":
        args.config = "config/ppo_caps.yaml"
    elif args.env_id == "ACNoVa-v0" or args.env_id == "ACNoVaIntegErr-v0":
        args.config = "config/ppo_caps_no_va.yaml"

    run_name = f"ppo_{args.exp_name}_{args.seed}_{strftime('%d-%m_%H:%M:%S', localtime())}"

    save_path: str = "models/train/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.track:
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
    envs = gym.vector.SyncVectorEnv(
        [ppo.make_env(args.env_id, args.config, "none", None, eval=False, gamma=args.gamma) for i in range(args.num_envs)]
    )
    unwr_envs = [envs.envs[i].unwrapped for i in range(args.num_envs)]

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = ppo.Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    trim_point: TrimPoint = TrimPoint(aircraft_id='x8')
    if args.env_id == "AttitudeControl-v0":
        trim_acts = torch.tensor([trim_point.aileron, trim_point.elevator, trim_point.throttle]).to(device)
    elif args.env_id == "ACNoVa-v0" or args.env_id == "ACNoVaIntegErr-v0":
        trim_acts = torch.tensor([trim_point.aileron, trim_point.elevator]).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs_t1 = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
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
    next_obs, _ = envs.reset(options=sim_options)
    next_obs = torch.Tensor(next_obs).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Generate a reference sequence and sample the first steps
    refSeqs = [RefSequence(num_refs=5) for _ in range(args.num_envs)]
    for _ in range(args.num_envs):
        refSeqs[_].sample_steps()

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            if args.track:
                wandb.log({"global_step": global_step})
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            terminateds[step] = next_terminated

            for i in range(args.num_envs):
                ith_env_step = unwr_envs[i].sim[unwr_envs[i].current_step]
                if args.rand_targets:
                    roll_ref, pitch_ref, airspeed_ref = refSeqs[i].sample_refs(ith_env_step, i)
                    if args.env_id == "AttitudeControl-v0":
                        unwr_envs[i].set_target_state(roll_ref, pitch_ref, airspeed_ref)
                    elif args.env_id == "ACNoVa-v0" or args.env_id == "ACNoVaIntegErr-v0":
                        unwr_envs[i].set_target_state(roll_ref, pitch_ref)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_mean, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            truncateds[step] = torch.Tensor(truncated).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_terminated = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)

            dones = np.logical_or(terminated, truncated)

            for env_i, done in enumerate(dones):
                if done:
                    obs_t1[step][env_i] = obs[step][env_i]
                    refSeqs[env_i].sample_steps() # Sample a new sequence of reference steps
                else:
                    obs_t1[step][env_i] = next_obs[env_i]

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']} \n" + \
                      f"episode_end={info['episode_end']}, out_of_bounds={info['out_of_bounds']}\n********")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                r_per_step = info["episode"]["r"]/info["episode"]["l"]
                writer.add_scalar("charts/reward_per_step", r_per_step, global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminateds[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.transpose(0,1).reshape((-1,) + envs.single_observation_space.shape)
        b_obs_t1 = obs_t1.transpose(0, 1).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.transpose(0, 1).reshape(-1)
        b_actions = actions.transpose(0, 1).reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.transpose(0, 1).reshape(-1)
        b_returns = returns.transpose(0, 1).reshape(-1)
        b_values = values.transpose(0, 1).reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, __, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # CAPS losses
                # Temporal Smoothing
                act_means = agent.get_action_and_value(b_obs[mb_inds])[1]
                next_act_means = agent.get_action_and_value(b_obs_t1[mb_inds])[1]
                ts_loss = torch.linalg.norm(act_means - next_act_means, ord=2)

                # Spatial Smoothing
                state_problaw = Normal(b_obs[mb_inds], 0.01)
                state_sampled = state_problaw.sample()
                act_means_bar = agent.get_action_and_value(state_sampled)[1]
                ss_loss = torch.linalg.norm(act_means - act_means_bar, ord=2)

                # preactivation loss
                pa_loss = torch.linalg.norm(act_means - trim_acts, ord=2)

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.ts_coef * ts_loss + args.ss_coef * ss_loss + args.pa_coef * pa_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        action_std = agent.get_action_std()[0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/value_loss_term", args.vf_coef * v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/ts", ts_loss.item(), global_step)
        writer.add_scalar("losses/ts_term", args.ts_coef * ts_loss.item(), global_step)
        writer.add_scalar("losses/ss", ss_loss.item(), global_step)
        writer.add_scalar("losses/ss_term", args.ss_coef * ss_loss.item(), global_step)
        writer.add_scalar("losses/pa", ss_loss.item(), global_step)
        writer.add_scalar("losses/pa_term", args.pa_coef * pa_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_term", args.ent_coef * entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("action_std/da", action_std[0], global_step)
        writer.add_scalar("action_std/de", action_std[1], global_step)
        if args.env_id == "AttitudeControl-v0":
            writer.add_scalar("action_std/dt", action_std[2], global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    # Evaluate the agent
    if not args.no_eval:
        print("******** Evaluating agent... ***********")
        e_env = envs.envs[0]
        e_env.eval = True
        telemetry_file = f"telemetry/{run_name}.csv"
        e_obs, _ = e_env.reset(options={"render_mode": "log"})
        e_env.unwrapped.telemetry_setup(telemetry_file)
        e_obs = torch.Tensor(e_obs).unsqueeze(0).to(device)
        e_refSeq = RefSequence(num_refs=5)
        e_refSeq.sample_steps()
        for step in range(6000):
            roll_ref, pitch_ref, airspeed_ref = e_refSeq.sample_refs(step)
            if args.env_id == "AttitudeControl-v0":
                e_env.unwrapped.set_target_state(roll_ref, pitch_ref, airspeed_ref)
            elif args.env_id == "ACNoVa-v0" or args.env_id == "ACNoVaIntegErr-v0":
                e_env.unwrapped.set_target_state(roll_ref, pitch_ref)

            action = agent.get_action_and_value(e_obs)[1][0].detach().cpu().numpy()
            e_obs, reward, truncated, terminated, info = e_env.step(action)
            e_obs = torch.Tensor(e_obs).unsqueeze(0).to(device)
            done = np.logical_or(truncated, terminated)

            if done:
                e_refSeq.sample_steps(offset=step)
                print(f"Episode reward: {info['episode']['r']}")
                r_per_step = info["episode"]["r"]/info["episode"]["l"]
                # Save the best agents depending on the env
                if args.save_best:
                    if (args.env_id == "AttitudeControl-v0" and r_per_step > -0.20) or \
                       ((args.env_id == "ACNoVa-v0" or args.env_id == "ACNoVaIntegErr-v0") and r_per_step > -0.06):
                        save_model(save_path, run_name, agent, e_env, args.seed)
                else:
                    save_model(save_path, run_name, agent, e_env, args.seed)
                break
        telemetry_df = pd.read_csv(telemetry_file)
        telemetry_table = wandb.Table(dataframe=telemetry_df)
        wandb.log({"telemetry": telemetry_table})
    # Even if we don't evaluate, we still want to save the model
    else:
        save_model(save_path, run_name, agent, envs.envs[0], args.seed)

    envs.close()
    writer.close()