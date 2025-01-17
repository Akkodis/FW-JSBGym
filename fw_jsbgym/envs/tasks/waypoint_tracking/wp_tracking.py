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


class WaypointTracking(JSBSimEnv):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z).
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.enu_x_m, prp.enu_y_m, prp.enu_z_m, # position
            prp.enu_x_err_m, prp.enu_y_err_m, prp.enu_z_err_m, # position error
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps = (
            prp.target_enu_x_m, prp.target_enu_y_m, prp.target_enu_z_m # target position
        )

        self.error_prps = (
            prp.enu_x_err_m, prp.enu_y_err_m, prp.enu_z_err_m # position error
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation_deque: Deque[np.ndarray] = deque(maxlen=self.task_cfg.mdp.obs_hist_size) # deque of 1D nparrays containing self.State

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()


        self.dist_to_target = np.inf
        self.dist_to_target_prev = np.inf
        self.prev_target_x = 0.0
        self.prev_target_y = 0.0
        self.prev_target_z = 0.0

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def observe_state(self, first_obs: bool = False) -> np.ndarray:
        # observe the state (actualizes self.state)
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


    def set_target_state(self, target_enu_x_m, target_enu_y_m, target_enu_z_m) -> None:
        if target_enu_x_m != self.prev_target_x:
            print("Target X changed to: ", target_enu_x_m)
        if target_enu_y_m != self.prev_target_y:
            print("Target Y changed to: ", target_enu_y_m)
        if target_enu_z_m != self.prev_target_z:
            print("Target Z changed to: ", target_enu_z_m)

        self.sim[prp.target_enu_x_m] = target_enu_x_m
        self.sim[prp.target_enu_y_m] = target_enu_y_m
        self.sim[prp.target_enu_z_m] = target_enu_z_m

        self.prev_target_x = target_enu_x_m
        self.prev_target_y = target_enu_y_m
        self.prev_target_z = target_enu_z_m

        self.target = self.TargetState(*[self.sim[prop] for prop in self.target_prps])


    def update_errors(self):
        """
            Updates the errors based on the current state.
        """
        # update error jsbsim properties
        self.sim[prp.enu_x_err_m] = self.sim[prp.target_enu_x_m] - self.sim[prp.enu_x_m]
        self.sim[prp.enu_y_err_m] = self.sim[prp.target_enu_y_m] - self.sim[prp.enu_y_m]
        self.sim[prp.enu_z_err_m] = self.sim[prp.target_enu_z_m] - self.sim[prp.enu_z_m]
        # print('----------------------------------')
        # print("Curr   Z: ", self.sim[prp.enu_z_m])
        # print("Target Z: ", self.sim[prp.target_enu_z_m])
        # print("Error  Z: ", self.sim[prp.enu_z_err_m])
        # print('Current X: ', self.sim[prp.enu_x_m])
        # print('Target  X: ', self.sim[prp.target_enu_x_m])
        # print('Error   X: ', self.sim[prp.enu_x_err_m])
        # print('Current Y: ', self.sim[prp.enu_y_m])
        # print('Target  Y: ', self.sim[prp.target_enu_y_m])
        # print('Error   Y: ', self.sim[prp.enu_y_err_m])

        # update the error namedtuple
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_prps])


    def reset_target_state(self) -> None:
        """
            Resets the target state to the current state and the PID controller.
        """
        self.set_target_state(self.sim[prp.lat_gc_deg], self.sim[prp.lng_gc_deg], self.sim[prp.altitude_sl_m])


    def get_reward(self, action: np.ndarray) -> float:
        """
            Reward function for the waypoint tracking task.
        """
        self.dist_to_target = np.sqrt(self.sim[prp.enu_x_err_m]**2 + self.sim[prp.enu_y_err_m]**2 + self.sim[prp.enu_z_err_m]**2)
        d_rate = 0.0
        r_dist = 0.0
        r_reached = 0.0
        if np.any(np.isinf(self.dist_to_target_prev + self.dist_to_target)):
            d_rate = 0.0
        else:
            d_rate = self.dist_to_target_prev - self.dist_to_target
            # r_dist = 0.1*self.dist_to_target - max(3*d_rate, 0.0)
            r_dist = max(3*d_rate, 0.0) + min(3*d_rate, 0.0)
            # if d_rate > 0:
            #     r_dist = d_rate
            # else:
            #     r_dist = -1
        self.dist_to_target_prev = self.dist_to_target


        if self.dist_to_target < 3:
            r_reached = 30.0

        r_total = (r_dist + r_reached)

        # r_x = self.sim[prp.enu_x_err_m] / 300
        # r_y = self.sim[prp.enu_y_err_m]
        # d_rate = self.dist_to_target_prev - r_y
        # if d_rate > 0:
        #     r_y = 1
        # else:
        #     r_y = -1
        # self.dist_to_target_prev = r_y
        # # r_z = self.sim[prp.enu_z_err_m] / 100
        # # r_total = -(r_x + r_y + r_z)
        # r_total = r_y

        # if self.sim[prp.enu_y_err_m] < 10:
        #     print("Reached y target")
        #     r_total = 300.0


        # populate reward properties
        # self.sim[prp.reward_enu_x] = r_x
        # self.sim[prp.reward_enu_y] = r_y
        # self.sim[prp.reward_enu_z] = r_z
        self.sim[prp.reward_wp_total] = r_total

        return r_total


    def is_terminated(self):
        return self.dist_to_target < 3



class WaypointTrackingNoVa(WaypointTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z).
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # PI controller for airspeed
        self.pid_airspeed = PID(kp=0.5, ki=0.1, kd=0.0,
                           dt=self.fdm_dt, trim=TrimPoint(), 
                           limit=AeroModel().throttle_limit, is_throttle=True
        )
        self.pid_airspeed.set_reference(60)

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def apply_action(self, action: np.ndarray):
        super().apply_action(action)

        if action.shape != self.action_space.shape:
            raise ValueError(f"Action shape {action.shape} is not compatible with action space {self.action_space.shape}")

        throttle_cmd, airspeed_err, _ = self.pid_airspeed.update(state=self.sim[prp.airspeed_kph], saturate=True)
        self.sim[prp.throttle_cmd] = throttle_cmd


    def reset_target_state(self) -> None:
        """
            Resets the target state to the current state and the PID controller.
        """
        super().reset_target_state()
        self.pid_airspeed.reset()


class AltitudeTracking(JSBSimEnv):
    def __init__(self, cfg_env, telemetry_file: str='', render_mode: str='none'):
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        # SS 1
        # self.state_prps: Tuple[BoundedProperty] = (
        #     prp.enu_z_err_m, 
        #     prp.roll_rad, prp.pitch_rad,
        #     prp.p_radps, prp.q_radps,
        #     prp.alpha_rad, prp.beta_rad,
        #     prp.airspeed_kph,
        #     prp.elevator_cmd, prp.throttle_cmd
        # )

        # SS 2
        # self.state_prps: Tuple[BoundedProperty] = (
        #     prp.enu_z_err_m,
        #     prp.u_fps, prp.w_fps,
        #     prp.pitch_rad,
        #     prp.q_radps,
        #     prp.alpha_rad, prp.beta_rad,
        #     prp.airspeed_kph,
        #     prp.elevator_cmd, prp.throttle_cmd
        # )

        # SS 3
        self.state_prps: Tuple[BoundedProperty] = (
            prp.enu_z_err_m,
            prp.roll_rad, prp.pitch_rad,
            prp.airspeed_kph,
            prp.p_radps, prp.q_radps, prp.r_radps,
            prp.alpha_rad, prp.beta_rad,
            prp.elevator_cmd, prp.throttle_cmd
        )

        self.action_prps: Tuple[BoundedProperty] = (
            prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps: Tuple[BoundedProperty] = (
            prp.target_enu_z_m,
        )

        self.error_prps: Tuple[BoundedProperty] = (
            prp.enu_z_err_m,
        )

        self.reward_prps: Tuple[BoundedProperty] = (
            prp.reward_total,
        )

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.prev_target_z = 0.0

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def reset_target_state(self):
        self.set_target_state(np.array([self.sim[prp.enu_z_m]]))


    def set_target_state(self, target_state:np.ndarray):
        """
            Sets the target state of the task.
            Args: target_state: np.ndarray of shape (1,) with the target altitude.
        """
        # check that the target state has the correct shape
        if target_state.shape[0] != len(self.target_prps):
            raise ValueError(f"Target state should be a 1D ndarray of length {len(self.target_prps)} but got shape {target_state.shape}")

        if target_state[0] != self.prev_target_z:
            print("Target Z changed to: ", target_state[0])
        self.sim[prp.target_enu_z_m] = target_state[0]
        self.prev_target_z = target_state[0]
        self.target = self.TargetState(*[self.sim[prop] for prop in self.target_prps])


    def update_errors(self):
        self.sim[prp.enu_z_err_m] = self.sim[prp.target_enu_z_m] - self.sim[prp.enu_z_m]
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_prps])


    def apply_action(self, action):
        super().apply_action(action)
        self.sim[prp.aileron_cmd] = TrimPoint().aileron


    def get_reward(self, action):
        r_dist = np.abs(self.sim[prp.enu_z_err_m])
        r_dist = 10 * np.tanh(0.005 * r_dist)
        r_total = -r_dist
        self.sim[prp.reward_total] = r_total
        return r_total

