import gymnasium as gym
import numpy as np

from omegaconf import DictConfig
from typing import Tuple, Deque, Dict
from collections import deque

from fw_jsbgym.envs.jsbsim_env import JSBSimEnv
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty

from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel


class AltitudeTracking(JSBSimEnv):
    """
        Altitude Tracking task.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.state_prps = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd
        )

        self.target_prps = (
            prp.target_lat_deg, prp.target_lon_deg, prp.target_alt_m
        )

        self.error_prps = (
            prp.lat_err, prp.lon_err, prp.alt_err
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation_deque: Deque[np.ndarray] = deque(maxlen=self.task_cfg.mdp.obs_hist_size) # deque of 1D nparrays containing self.State

        # declaring action history. Deque with a maximum length of act_history_size
        self.action_hist: Deque[np.ndarray] = deque(maxlen=self.task_cfg.mdp.act_hist_size) # action type: np.ndarray

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # PI controller for airspeed
        self.pid_airspeed = PID(kp=0.5, ki=0.1, kd=0.0,
                           dt=self.fdm_dt, trim=TrimPoint(), 
                           limit=AeroModel().throttle_limit, is_throttle=True
        ) 

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def apply_action(self, action: np.ndarray):
        super().apply_action(action)

        if action.shape != self.action_space.shape:
            raise ValueError(f"Action shape {action.shape} is not compatible with action space {self.action_space.shape}")

        self.pid_airspeed.set_reference(60)
        throttle_cmd, airspeed_err, _ = self.pid_airspeed.update(state=self.sim[prp.airspeed_kph], saturate=True)
        self.sim[prp.throttle_cmd] = throttle_cmd


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return super().step(action)
    

    def reset(self) -> np.ndarray:
        return super().reset()


    def observe_state(self, first_obs: bool = False) -> np.ndarray:
        super().observe_state()

        # if it's the first observation i.e. following a reset(): fill observation with obs_history_size * state
        if first_obs:
            for _ in range(self.task_cfg.mdp.obs_hist_size):
                self.observation_deque.append(self.state)
        # else just append the newest state
        else:
            self.observation_deque.append(self.state)

        # return observation as a numpy array and add one channel dim for CNN policy
        if self.task_cfg.mdp.obs_is_matrix:
            obs: np.ndarray = np.expand_dims(np.array(self.observation_deque), axis=0).astype(np.float32)
        else: # else return observation as a vector for MLP policy
            obs: np.ndarray = np.array(self.observation_deque).squeeze().flatten().astype(np.float32)
        return obs