import argparse
import gymnasium as gym
import numpy as np

from agents.pid import PID
from models import aerodynamics
from trim.trim_point import TrimPoint
from utils.eval_utils import RefSequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControlNoVa-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument("--tele-file", type=str, default="telemetry/pid_eval_telemetry.csv", 
        help="telemetry csv file")
    parser.add_argument('--rand-targets', action='store_true', help='set targets randomly')
    parser.add_argument('--rand-atmo-mag', action='store_true', help='randomize the wind and turb magnitudes at each episode')
    parser.add_argument('--turb', action='store_true', help='add turbulence')
    parser.add_argument('--wind', action='store_true', help='add wind')
    args = parser.parse_args()
    return args


def rearrange_obs(obs: np.ndarray) -> tuple[float, float, float, float, float]:
    roll = obs[0][4][0]
    pitch = obs[0][4][1]
    Va = obs[0][4][2]
    roll_rate = obs[0][4][3]
    pitch_rate = obs[0][4][4]
    return Va, roll, pitch, roll_rate, pitch_rate


if __name__ == '__main__':
    args = parse_args()
    if args.env_id == "AttitudeControl-v0":
        args.config = "config/ppo_caps.yaml"
    elif args.env_id == "AttitudeControlNoVa-v0":
        args.config = "config/ppo_caps_no_va.yaml"

    env = gym.make(args.env_id, config_file=args.config, render_mode=args.render_mode,
                    telemetry_file=args.tele_file)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    sim_options = {"atmosphere": {"rand_magnitudes": args.rand_atmo_mag, 
                                  "wind": args.wind,
                                  "turb": args.turb}}
    obs, _ = env.reset(options=sim_options)
    Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)
    refSeq = RefSequence(num_refs=5)

    x8 = aerodynamics.AeroModel()
    trim_point = TrimPoint('x8')
    # setting handtunes PID gains
    # lateral dynamics
    kp_roll: float = 1.0
    ki_roll: float = 0.0
    kd_roll: float = 0.5
    roll_pid = PID(
        kp=kp_roll, ki=ki_roll, kd=kd_roll,
        dt=env.sim.fdm_dt, limit=x8.aileron_limit
    )

    # longitudinal dynamics
    kp_pitch: float = -4.0
    ki_pitch: float = -0.75
    kd_pitch: float = -0.1
    pitch_pid = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                    dt=env.sim.fdm_dt, limit=x8.aileron_limit)

    kp_airspeed: float = 0.5
    ki_airspeed: float = 0.1
    kd_airspeed: float = 0.0
    airspeed_pid = PID(
        kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
        dt=env.sim.fdm_dt, trim=trim_point,
        limit=x8.throttle_limit, is_throttle=True
    )

    # set default target values
    roll_ref: float = 0.0
    pitch_ref: float = 0.0
    airspeed_ref: float = trim_point.Va_kph

    ref_data = np.load("ref_seq_arr.npy")

    for step in range(8000):
        # set random target values
        if args.rand_targets:
            # roll_ref, pitch_ref, airspeed_ref = refSeq.sample_refs(step)
            refs = ref_data[step]
            roll_ref, pitch_ref, airspeed_ref = refs[0], refs[1], refs[2]

        # apply target values
        roll_pid.set_reference(roll_ref)
        pitch_pid.set_reference(pitch_ref)
        if args.env_id == "AttitudeControl-v0":
            env.set_target_state(roll_ref, pitch_ref, airspeed_ref)
            airspeed_pid.set_reference(airspeed_ref)
            throttle_cmd, airspeed_err = airspeed_pid.update(state=Va, saturate=True)
        elif args.env_id == "AttitudeControlNoVa-v0":
            env.set_target_state(roll_ref, pitch_ref)

        elevator_cmd, pitch_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
        aileron_cmd, roll_err = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)

        if args.env_id == "AttitudeControl-v0":
            action = np.array([aileron_cmd, elevator_cmd, throttle_cmd])
        elif args.env_id == "AttitudeControlNoVa-v0":
            action = np.array([aileron_cmd, elevator_cmd])
        obs, reward, truncated, terminated, info = env.step(action)
        Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

        done = np.logical_or(truncated, terminated)
        if done:
            print(f"Episode reward: {info['episode']['r']}")
            # print(step)
            # refSeq.sample_steps(offset=step)
