import numpy as np
import yaml
import os

from typing import Tuple
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty
from fw_jsbgym.envs.jsbsim_env import JSBSimEnv


class SimpleAC_OMAC(JSBSimEnv):
    """
        Simple Aircraft Attitude Controller (AC) environment for JSBSim with states for OMAC algo reimplementation
    """
    def __init__(self, config_file="config.yaml", telemetry_file: str = '', render_mode: str = 'none'):
        print("current dir: ", os.getcwd())
        with open(config_file, "r") as file:
            cfg_all: dict = yaml.safe_load(file)
        super().__init__(jsbsim_config=cfg_all['JSBSimEnv'], 
                         telemetry_file=telemetry_file,
                         render_mode=render_mode,
                         aircraft_id='x8')

        # u, v, w, p, q, r, phi, theta, psi, (lat, lon, alt)
        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            # prp.roll_err, prp.pitch_err, prp.airspeed_err, # errors
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.aileron_cmd, prp.elevator_cmd, # control surface commands normalized [-1, 1]
            prp.throttle_cmd # throttle command normalized [0, 1]
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            # prp.target_roll_rad, prp.target_pitch_rad, # target attitude
            # prp.target_airspeed_kph # target airspeed
        )

        self.telemetry_prps: Tuple[BoundedProperty, ...] = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates and airspeed
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, # control surface commands
            prp.reward_total, # rewards
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps # target state variables

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, prp.airspeed_err # errors
        )

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.init()
        self.telemetry_setup(self.telemetry_file)


    def reset(self, seed: int=None, options: dict=None) -> np.ndarray:
        super().reset(seed=seed, options=options)

        self.reset_props()
        self.observation: np.ndarray = self.observe_state()

        info: dict = {"non_norm_obs": self.observation}
        self.render()
        return self.observation, info


    def step(self, action: np.ndarray):
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid")

        super().step(action)
        self.apply_action(action)

        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()
            self.sim[self.steps_left] -= 1
            self.sim[self.current_step] += 1

        # to be implemented
        # self.update_errors()

        # get the state
        self.observation = self.observe_state()

        # get the reward
        self.reward = self.get_reward()
        terminated = self.is_terminated()
        truncated, episode_end, out_of_bounds = self.is_truncated()
        
        # write telemetry to a csv file every agent step
        self.telemetry_logging()

        # info dict for debugging and misc infos
        info: dict = {"steps_left": self.sim[self.steps_left],
                      "non_norm_obs": self.observation,
                      "non_norm_reward": self.reward,
                      "episode_end": episode_end,
                      "out_of_bounds": out_of_bounds,
                    }

        return self.observation, self.reward, terminated, truncated, info


    # Placeholder reward function, returns 1
    def get_reward(self) -> float:
        r_total = 1.0
        self.sim[prp.reward_total] = r_total
        return r_total
