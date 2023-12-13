import argparse
import random
import torch
import numpy as np
from tqdm import tqdm

from agents import ppo
from trim.trim_point import TrimPoint
from utils.eval_utils import RefSequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControl-v0", 
        help="the id of the environment")
    parser.add_argument('--train-model', type=str, required=True, 
        help='agent model file name')
    parser.add_argument('--render-mode', type=str, 
        choices=['none','plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
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

    sim_options = {"atmosphere": {"rand_magnitudes": args.rand_atmo_mag, 
                                  "wind": args.wind,
                                  "turb": args.turb},
                   "seed": seed}

    # Generating a reference sequence
    # refSeq = RefSequence(num_refs=5)
    # refSeq.sample_steps()

    train_dict = torch.load(args.train_model, map_location=device)

    # setting the observation normalization parameters
    env.set_obs_rms(train_dict['norm_obs_rms']['mean'], 
                             train_dict['norm_obs_rms']['var'])

    # loading the agent
    ppo_agent = ppo.Agent(env).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()

    # set default target values
    roll_ref: float = 0.0
    pitch_ref: float = 0.0
    airspeed_ref: float = trim_point.Va_kph

    # load the reference sequence and initialize the evaluation arrays
    ref_data = np.load("ref_seq_arr.npy")
    e_actions = np.ndarray((ref_data.shape[0], env.action_space.shape[0]))
    e_obs = np.ndarray((ref_data.shape[0], env.observation_space.shape[2]))

    # start the environment
    obs, _ = env.reset(options=sim_options)
    obs = torch.Tensor(obs).unsqueeze_(0).to(device)

    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if args.render_mode == "none":
        total_steps = ref_data.shape[0]
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 8000

    for step in tqdm(range(total_steps)):
        if args.rand_targets:
            # roll_ref, pitch_ref, airspeed_ref = refSeq.sample_refs(step)
            refs = ref_data[step]
            roll_ref, pitch_ref = refs[0], refs[1]

        # if args.env_id == "AttitudeControl-v0":
        #     unwrapped_env.set_target_state(roll_ref, pitch_ref, airspeed_ref)
        if args.env_id == "AttitudeControlNoVa-v0":
            env.set_target_state(roll_ref, pitch_ref)

        action = ppo_agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
        e_actions[step] = action
        obs, reward, truncated, terminated, info = env.step(action)
        e_obs[step] = info["non_norm_obs"][0, -1]
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)

        done = np.logical_or(truncated, terminated)
        if done:
            print(f"Episode reward: {info['episode']['r']}")
            # refSeq.sample_steps(offset=step)

    env.close()

    # compute mean square error
    # Roll MSE
    roll_errors = e_obs[:, 6]
    roll_mse = np.mean(np.square(roll_errors))
    print(f"roll mse: {roll_mse}") # roll mse: 0.1750963732741717

    # Pitch MSE
    pitch_errors = e_obs[:, 7]
    pitch_mse = np.mean(np.square(pitch_errors))
    print(f"pitch mse: {pitch_mse}") # pitch mse: 0.06732408213127292