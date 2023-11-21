import argparse
import gymnasium as gym
import numpy as np

from agents.pid import PID
from models import aerodynamics
from trim.trim_point import TrimPoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo_caps.yaml",
        help="the config file of the environnement")
    parser.add_argument("--env-id", type=str, default="AttitudeControlTaskEnv-v0", 
        help="the id of the environment")
    parser.add_argument('--render-mode', type=str, 
        choices=['plot_scale', 'plot', 'fgear', 'fgear_plot', 'fgear_plot_scale'],
        help='render mode')
    parser.add_argument('--rand-targets', action='store_true', help='set targets randomly')
    parser.add_argument('--turb', action='store_true', help='add turbulence')
    parser.add_argument('--wind', action='store_true', help='add wind')
    args = parser.parse_args()
    return args


def rearrange_obs(obs: np.ndarray) -> tuple[float, float, float, float, float]:
    Va = obs[0][4][0]
    roll = obs[0][4][1]
    pitch = obs[0][4][2]
    roll_rate = obs[0][4][3]
    pitch_rate = obs[0][4][4]
    return Va, roll, pitch, roll_rate, pitch_rate


if __name__ == '__main__':
    args = parse_args()

    env = gym.make(args.env_id, config_file=args.config, render_mode=args.render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs, _ = env.reset()
    Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

    if args.turb:
        env.sim['atmosphere/turb-type'] = 3
        env.sim['atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps'] = 75
        env.sim["atmosphere/turbulence/milspec/severity"] = 6

    if args.wind:
        env.sim["atmosphere/wind-north-fps"] = 16.26 * 3.281 # mps to fps
        env.sim["atmosphere/wind-east-fps"] = 16.26 * 3.281 # mps to fps

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
    airspeed_ref: float = trim_point.Va_ms

    for step in range(2500):
        # set random target values
        if args.rand_targets and step % 500 == 0:
            roll_ref = np.random.uniform(-45, 45) * (np.pi / 180)
            pitch_ref = np.random.uniform(-15, 15) * (np.pi / 180)
            airspeed_ref = np.random.uniform(trim_point.Va_ms - 2, trim_point.Va_ms + 2)

        # apply target values
        env.set_target_state(airspeed_ref, roll_ref, pitch_ref)
        roll_pid.set_reference(roll_ref)
        pitch_pid.set_reference(pitch_ref)
        airspeed_pid.set_reference(airspeed_ref)

        throttle_cmd, airspeed_err = airspeed_pid.update(state=Va, saturate=True)
        elevator_cmd, pitch_err = pitch_pid.update(state=pitch, state_dot=pitch_rate, saturate=True, normalize=True)
        aileron_cmd, roll_err = roll_pid.update(state=roll, state_dot=roll_rate, saturate=True, normalize=True)

        action = np.array([elevator_cmd, aileron_cmd, throttle_cmd])
        obs, reward, truncated, terminated, info = env.step(action)
        Va, roll, pitch, roll_rate, pitch_rate = rearrange_obs(obs)

        done = np.logical_or(truncated, terminated)
        if done:
            print(f"Episode reward: {info['episode']['r']}")
            break 
