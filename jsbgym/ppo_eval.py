import argparse
import random
import torch
import numpy as np
from tqdm import tqdm

from agents import ppo
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.eval import metrics


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
    parser.add_argument('--save-res-file', action='store_true',default=False, help='save results to file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    np.set_printoptions(precision=3)
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
    ref_data = np.load("eval/ref_seq_arr.npy")
    ref_steps = np.load("eval/step_seq_arr.npy")


    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if args.render_mode == "none":
        total_steps = ref_data.shape[0]
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
                       }
                   }}

    if args.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [args.severity]

    pitch_mse_all = np.zeros(len(severity_range))
    roll_mse_all = np.zeros(len(severity_range))

    all_metrics = []

    for i, severity in enumerate(severity_range):
        sim_options["atmosphere"]["severity"] = severity
        e_actions = np.ndarray((total_steps, env.action_space.shape[0]))
        e_obs = np.ndarray((total_steps, env.observation_space.shape[2]))
        print(f"********** PPO METRICS {severity} **********")
        obs, _ = env.reset(options=sim_options)
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        for step in tqdm(range(total_steps)):
            if args.rand_targets:
                # roll_ref, pitch_ref, airspeed_ref = refSeq.sample_refs(step)
                refs = ref_data[step]
                roll_ref, pitch_ref = refs[0], refs[1]
                env.set_target_state(roll_ref, pitch_ref)

            action = ppo_agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
            e_actions[step] = action
            obs, reward, truncated, terminated, info = env.step(action)
            e_obs[step] = info["non_norm_obs"][0, -1]
            obs = torch.Tensor(obs).unsqueeze_(0).to(device)

            done = np.logical_or(truncated, terminated)
            if done:
                print(f"Episode reward: {info['episode']['r']}")
                obs, _ = env.reset()
                obs = torch.Tensor(obs).unsqueeze_(0).to(device)
                # refSeq.sample_steps(offset=step)
        all_metrics.append({severity: metrics.compute_all_metrics(e_obs, e_actions, ref_steps)})

    for sev_dict in all_metrics:
        for sev_name, sev_metrics in sev_dict.items():
            print(f"\nSeverity: {sev_name}")
            for name, value in sev_metrics.items():
                if isinstance(value, np.ndarray):
                    if value.shape[0] == 2: # if the metric has 2 fields: contains roll and pitch
                        print(f"  {name}:\n"
                            f"    roll: {value[0]}\n"
                            f"    pitch: {value[1]}")
                    elif value.shape[0] == 3: # if the metric has 3 fields: contains r, p, y angular vels
                        print(f"  {name}:\n"
                            f"    r: {value[0]}\n"
                            f"    p: {value[1]}\n"
                            f"    y: {value[2]}")
                else:
                    print(f"  {name}: {value}")

    if args.save_res_file:
        with open("eval/outputs/metrics_ppo.txt", "w") as f:
            for sev_dict in all_metrics:
                for sev_name, sev_metrics in sev_dict.items():
                    f.write(f"\nSeverity: {sev_name}\n")
                    for name, value in sev_metrics.items():
                        if isinstance(value, np.ndarray):
                            if value.shape[0] == 2: # if the metric has 2 fields: contains roll and pitch
                                f.write(f"  {name}:\n"
                                    f"    roll: {value[0]}\n"
                                    f"    pitch: {value[1]}\n")
                            elif value.shape[0] == 3: # if the metric has 3 fields: contains r, p, y angular vels
                                f.write(f"  {name}:\n"
                                    f"    r: {value[0]}\n"
                                    f"    p: {value[1]}\n"
                                    f"    y: {value[2]}\n")
                        else:
                            f.write(f"  {name}: {value}\n")

    # np.save("eval/e_ppo_obs.npy", e_obs)
    # np.save("eval/e_ppo_actions.npy", e_actions)
    env.close()
