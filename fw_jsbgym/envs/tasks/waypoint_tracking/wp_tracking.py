import gymnasium as gym
import numpy as np

from omegaconf import DictConfig
from typing import Tuple, Dict

from fw_jsbgym.envs.tasks.jsbsim_task import JSBSimTask
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty
from fw_jsbgym.utils import conversions

from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel


class WaypointTracking(JSBSimTask):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z).
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.ecef_x_err_m, prp.ecef_y_err_m, prp.ecef_z_err_m, # position error
            prp.airspeed_kph, # airspeed
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps = (
            prp.target_ecef_x_m, prp.target_ecef_y_m, prp.target_ecef_z_m # target position
        )

        # ENU target position, for telemetry
        self.target_enu_prps = (
            prp.target_enu_e_m, prp.target_enu_n_m, prp.target_enu_u_m # target position in ENU
        )

        self.error_prps = (
            prp.ecef_x_err_m, prp.ecef_y_err_m, prp.ecef_z_err_m # position error
        )

        self.reward_prps = (
            prp.reward_total, prp.reward_dist, prp.reward_progress,
            prp.dist_to_target_m
        )

        self.enu_prps = (
            prp.enu_e_m, prp.enu_n_m, prp.enu_u_m, # position in ENU
        )
        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.enu_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.dist_to_target = 0.0
        self.prev_dist_to_target = 0.0
        self.prev_target_x = 0.0
        self.prev_target_y = 0.0
        self.prev_target_z = 0.0

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and WaypointTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # update the action history
        self.update_action_history(action)
        # step the simulation
        self.observation, self.reward, terminated, truncated, info = super().step(action)
        info["target_missed"] = self.inout_missed_sphere
        info["target_reached"] = self.target_reached
        info["success"] = int(info["target_missed"]) * 1 + int(info["target_reached"]) * 2
        # reset the flags if the episode is ending for safety and reset distance to target
        if terminated or truncated:
            self.target_reached = False
            self.in_missed_sphere = False
            self.inout_missed_sphere = False
        return self.observation, self.reward, terminated, truncated, info


    def observe_state(self, first_obs = False):
        # it's the first obs (a reset has been called) distance is 0.0
        # otherwise calculate the distance to the target
        if first_obs:
            self.dist_to_target = 0.0
            self.prev_dist_to_target = 0.0
        else:
            self.prev_dist_to_target = self.dist_to_target
            self.dist_to_target = np.sqrt(
                self.sim[prp.ecef_x_err_m]**2 + 
                self.sim[prp.ecef_y_err_m]**2 + 
                self.sim[prp.ecef_z_err_m]**2
            )

        self.sim[prp.dist_to_target_m] = self.dist_to_target
        # print(f"Distance to target: {self.dist_to_target:.3f} m")
        return super().observe_state(first_obs)


    def set_target_state(self, target: np.ndarray) -> None:
        target_ecef_x_m, target_ecef_y_m, target_ecef_z_m = target
        if np.any(target != [self.prev_target_x, self.prev_target_y, self.prev_target_z]):
            target_enu = conversions.ecef2enu(target_ecef_x_m, target_ecef_y_m, target_ecef_z_m,
                                                self.sim[prp.ic_lat_gd_deg], self.sim[prp.ic_long_gc_deg], 0.0)
            print("-- SETTING TARGET --")
            print(f"Target (ENU) x: {target_enu[0]:.3f} y: {target_enu[1]:.3f} z: {target_enu[2]:.3f}")

        self.sim[prp.target_ecef_x_m] = target_ecef_x_m
        self.sim[prp.target_ecef_y_m] = target_ecef_y_m
        self.sim[prp.target_ecef_z_m] = target_ecef_z_m

        self.prev_target_x = target_ecef_x_m
        self.prev_target_y = target_ecef_y_m
        self.prev_target_z = target_ecef_z_m


    def reset_target_state(self) -> None:
        """
            Resets the target state to the current state
        """
        print("--- RESETTING TARGET ---")
        init_target = np.array([self.sim[prp.ecef_x_m], self.sim[prp.ecef_y_m], self.sim[prp.ecef_z_m]])
        self.set_target_state(init_target)
        print("------------------------")


    def update_errors(self, first_err=False):
        """
            Updates the errors based on the current state.
        """
        # update error jsbsim properties
        self.sim[prp.ecef_x_err_m] = self.sim[prp.target_ecef_x_m] - self.sim[prp.ecef_x_m]
        self.sim[prp.ecef_y_err_m] = self.sim[prp.target_ecef_y_m] - self.sim[prp.ecef_y_m]
        self.sim[prp.ecef_z_err_m] = self.sim[prp.target_ecef_z_m] - self.sim[prp.ecef_z_m]


    def get_reward(self, action: np.ndarray) -> float:
        if self.task_cfg.reward.name == "wp_dist":
            return self.distance_reward(action)
        elif self.task_cfg.reward.name == "wp_prog":
            return self.progress_reward(action)
        else:
            raise ValueError(f"Unknown reward type: {self.task_cfg.reward.name}")


    # Distance based reward
    def distance_reward(self, action: np.ndarray) -> float:
        assert self.task_cfg.reward.name == "wp_dist"
        r_w: dict = self.task_cfg.reward.weights

        r_dist = r_w["r_dist"]["max_r"] * np.tanh(r_w["r_dist"]["tanh_scale"] * self.dist_to_target)
        self.sim[prp.reward_dist] = r_dist

        r_actvar = 0.0
        if r_w["act_var"]["enabled"]:
            mean_actvar = np.mean(np.abs(action - np.array(self.action_hist)[-2])) # normalized by distance between action limits
            r_actvar = r_w["act_var"]["max_r"] * np.tanh(r_w["act_var"]["tanh_scale"] * mean_actvar)
        self.sim[prp.reward_actvar] = r_actvar

        r_total = -(r_dist + r_actvar)
        self.sim[prp.reward_total] = r_total

        return r_total


    # Progress based reward
    def progress_reward(self, action: np.ndarray) -> float:
        assert self.task_cfg.reward.name == "wp_prog"
        r_w: dict = self.task_cfg.reward.weights
        r_progress = r_w["r_prog"]["scale"] * (self.prev_dist_to_target - self.dist_to_target)
        self.sim[prp.reward_progress] = r_progress
        self.sim[prp.reward_total] = r_progress
        return r_progress


    # mix of progress and distance
    # def get_reward(self, action: np.ndarray) -> float:
    #     # print(f"Prev dist: {self.prev_dist_to_target:.5f} Dist: {self.dist_to_target:.5f}")
    #     d_rate = 6.25 * (self.prev_dist_to_target - self.dist_to_target)
    #     r_progress = d_rate + 9 * np.exp(-0.1 * self.dist_to_target)
    #     # print(f"Progress: {r_progress:.5f}")
    #     self.sim[prp.reward_total] = r_progress
    #     return r_progress


    def is_waypoint_reached(self):
        """
            Returns True if the distance to the target is less than 3 meters.
        """
        self.target_reached = False
        if self.dist_to_target < 3:
        # if self.dist_to_target < 5:
            print(f"Target Reached! @ step : {self.sim[self.current_step]}")
            # resets the missed sphere flag since the target was reached and the episode is about to end
            self.in_missed_sphere = False
            self.target_reached = True
        return self.target_reached


    def is_waypoint_missed(self):
        """
            Returns True if the UAV missed the target.
        """
        # in -> out of the missed sphere is set to False by default
        self.inout_missed_sphere = False
        # UAV enters some sphere around the target
        if self.dist_to_target < 10.0:
            self.in_missed_sphere = True
        # UAV exits the sphere
        if self.in_missed_sphere and self.dist_to_target > 10.0:
            self.in_missed_sphere = False
            self.inout_missed_sphere = True
            print(f"Target Missed! @ step : {self.sim[self.current_step]}")
        return self.inout_missed_sphere


    def is_terminated(self):
        terminated = self.is_waypoint_reached()
        return terminated


    def is_truncated(self) -> Tuple[bool, Dict]:
        truncated, info_trunc = super().is_truncated()
        wp_missed = self.is_waypoint_missed()
        # add the waypoint missed flag to the info dictionary
        info_trunc["wp_missed"] = wp_missed
        # truncation occurs if we missed the waypoint or the episode is truncated (e.g. time limit or out of bounds)
        truncated = truncated or wp_missed
        return truncated, info_trunc
    
    def reset_ext_state_props(self):
        pass


class WaypointTrackingENU(WaypointTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x,y,z in ENU coordinates).
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.enu_e_err_m, prp.enu_n_err_m, prp.enu_u_err_m, # position error
            prp.airspeed_kph, # airspeed
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        # ENU target position, for telemetry
        self.target_prps = (
            prp.target_enu_e_m, prp.target_enu_n_m, prp.target_enu_u_m # target position in ENU
        )

        self.error_prps = (
            prp.enu_e_err_m, prp.enu_n_err_m, prp.enu_u_err_m # position error
        )

        self.reward_prps = (
            prp.reward_total,
            prp.reward_enu_e, prp.reward_enu_n, prp.reward_enu_u, 
            prp.dist_to_target_m
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps \
                            + self.error_prps + self.reward_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.dist_to_target = 0.0
        self.prev_dist_to_target = 0.0
        self.prev_target_x = 0.0
        self.prev_target_y = 0.0
        self.prev_target_z = 0.0

        self.prev_enu_e_err_m = 0.0
        self.prev_enu_n_err_m = 0.0
        self.prev_enu_u_err_m = 0.0

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and WaypointTrackingENU.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def observe_state(self, first_obs=False):
        # convert ecef to enu current UAV position
        enu_pos = conversions.ecef2enu(
            x=self.sim[prp.ecef_x_m], y=self.sim[prp.ecef_y_m], z=self.sim[prp.ecef_z_m],
            ref_lat=self.sim[prp.ic_lat_gd_deg],
            ref_lon=self.sim[prp.ic_long_gc_deg], 
            ref_alt=0.0,
        )
        self.sim[prp.enu_e_m] = enu_pos[0]
        self.sim[prp.enu_n_m] = enu_pos[1]
        self.sim[prp.enu_u_m] = enu_pos[2]

        if first_obs:
            self.dist_to_target = 0.0
            self.prev_dist_to_target = 0.0
        else:
            self.prev_dist_to_target = self.dist_to_target
            self.dist_to_target = np.sqrt(
                self.sim[prp.enu_e_err_m]**2 + 
                self.sim[prp.enu_n_err_m]**2 + 
                self.sim[prp.enu_u_err_m]**2
            )
        self.sim[prp.dist_to_target_m] = self.dist_to_target
        return super(WaypointTracking, self).observe_state(first_obs)


    def reset_ext_state_props(self):
        """
            Resets to the initial ENU position of the aircraft.
            Need to recompute it from ECEF position, since JSBSim does not provide ENU positions.
        """
        enu_pos = conversions.ecef2enu(
            x=self.sim[prp.ecef_x_m], y=self.sim[prp.ecef_y_m], z=self.sim[prp.ecef_z_m],
            ref_lat=self.sim[prp.ic_lat_gd_deg],
            ref_lon=self.sim[prp.ic_long_gc_deg], 
            ref_alt=0.0,
        )
        self.sim[prp.enu_e_m] = enu_pos[0]
        self.sim[prp.enu_n_m] = enu_pos[1]
        self.sim[prp.enu_u_m] = enu_pos[2]


    def set_target_state(self, target: np.ndarray) -> None:
        target_enu_e_m, target_enu_n_m, target_enu_u_m = target
        if np.any(target != [self.prev_target_x, self.prev_target_y, self.prev_target_z]):
            print("-- SETTING TARGET --")
            print(f"Target (ENU) E: {target_enu_e_m:.3f} N: {target_enu_n_m:.3f} U: {target_enu_u_m:.3f}")

        self.sim[prp.target_enu_e_m] = target_enu_e_m
        self.sim[prp.target_enu_n_m] = target_enu_n_m
        self.sim[prp.target_enu_u_m] = target_enu_u_m

        self.prev_target_x = target_enu_e_m
        self.prev_target_y = target_enu_n_m
        self.prev_target_z = target_enu_u_m


    def reset_target_state(self) -> None:
        """
            Resets the target state to the current state
        """
        print("--- RESETTING TARGET ---")
        init_target = np.array([self.sim[prp.enu_e_m], self.sim[prp.enu_n_m], self.sim[prp.enu_u_m]])
        self.set_target_state(init_target)
        print("------------------------")


    def update_errors(self, first_err=False):
        """
            Updates the errors based on the current state.
        """
        if first_err:
            self.prev_enu_e_err_m = 0.0
            self.prev_enu_n_err_m = 0.0
            self.prev_enu_u_err_m = 0.0
        else:
            self.prev_enu_e_err_m = self.sim[prp.enu_e_err_m]
            self.prev_enu_n_err_m = self.sim[prp.enu_n_err_m]
            self.prev_enu_u_err_m = self.sim[prp.enu_u_err_m]

        # update with newly computed errors
        self.sim[prp.enu_e_err_m] = self.sim[prp.target_enu_e_m] - self.sim[prp.enu_e_m]
        self.sim[prp.enu_n_err_m] = self.sim[prp.target_enu_n_m] - self.sim[prp.enu_n_m]
        self.sim[prp.enu_u_err_m] = self.sim[prp.target_enu_u_m] - self.sim[prp.enu_u_m]


    # Distance based reward but z and xy distances are weighted differently
    # def get_reward(self, action: np.ndarray) -> float:
    #     assert self.task_cfg.reward.name == "wp_dist_xyz" 
    #     r_w: dict = self.task_cfg.reward.weights

    #     x_abs_err = np.abs(self.sim[prp.enu_e_err_m])
    #     r_x = r_w["r_x"]["c_x"] * np.clip(x_abs_err / r_w["r_x"]["max_x"], 0.0, 1.0)
    #     self.sim[prp.reward_enu_e] = r_x

    #     y_abs_err = np.abs(self.sim[prp.enu_n_err_m])
    #     r_y = r_w["r_y"]["c_y"] * np.clip(y_abs_err / r_w["r_y"]["max_y"], 0.0, 1.0)
    #     self.sim[prp.reward_enu_n] = r_y

    #     z_abs_err = np.abs(self.sim[prp.enu_u_err_m])
    #     r_z = r_w["r_z"]["c_z"] * np.clip(z_abs_err / r_w["r_z"]["max_z"], 0.0, 1.0)
    #     self.sim[prp.reward_enu_u] = r_z

    #     r_total = -(r_x + r_y + r_z)
    #     self.sim[prp.reward_total] = r_total

    #     return r_total


    # Progress based reward with separated components
    def get_reward(self, action: np.ndarray) -> float:
        assert self.task_cfg.reward.name == "wp_prog_xyz"
        r_w: dict = self.task_cfg.reward.weights
        abs_x_err = np.abs(self.sim[prp.enu_e_err_m])
        abs_y_err = np.abs(self.sim[prp.enu_n_err_m])
        abs_z_err = np.abs(self.sim[prp.enu_u_err_m])
        abs_prev_x_err = np.abs(self.prev_enu_e_err_m)
        abs_prev_y_err = np.abs(self.prev_enu_n_err_m)
        abs_prev_z_err = np.abs(self.prev_enu_u_err_m)

        r_x_prog = r_w["r_x"]["scale"] * (abs_prev_x_err - abs_x_err)
        r_y_prog = r_w["r_y"]["scale"] * (abs_prev_y_err - abs_y_err)
        r_z_prog = r_w["r_z"]["scale"] * (abs_prev_z_err - abs_z_err)

        self.sim[prp.reward_enu_e] = r_x_prog
        self.sim[prp.reward_enu_n] = r_y_prog
        self.sim[prp.reward_enu_u] = r_z_prog

        r_progress = r_x_prog + r_y_prog + r_z_prog
        self.sim[prp.reward_total] = r_progress
        return r_progress



class WaypointVaTracking(WaypointTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z) in ECEF.
        It also has to do so while maintaining a constant given airspeed. (Here maintained at 60 kph)
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.ecef_x_err_m, prp.ecef_y_err_m, prp.ecef_z_err_m, # position error
            prp.airspeed_err_kph, # airspeed error
            prp.airspeed_kph, # airspeed
            prp.u_kph, prp.v_kph, prp.w_kph, # linear body velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps = (
            prp.target_ecef_x_m, prp.target_ecef_y_m, prp.target_ecef_z_m, # target position
            prp.target_airspeed_kph # target airspeed
        )

        self.error_prps = (
            prp.ecef_x_err_m, prp.ecef_y_err_m, prp.ecef_z_err_m, # position error
            prp.airspeed_err_kph # airspeed error
        )

        self.reward_prps = (
            prp.reward_total, prp.reward_progress,
            prp.reward_airspeed, prp.dist_to_target_m
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.prev_target_airspeed = 0.0

        if self.jsbsim_cfg.debug and WaypointVaTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def set_target_state(self, target: np.ndarray) -> None:
        target_ecef_x_m, target_ecef_y_m, target_ecef_z_m, target_airspeed_kph = target
        if np.any(target != [self.prev_target_x, self.prev_target_y, self.prev_target_z, self.prev_target_airspeed]):
            target_enu = conversions.ecef2enu(target_ecef_x_m, target_ecef_y_m, target_ecef_z_m,
                                                self.sim[prp.ic_lat_gd_deg], self.sim[prp.ic_long_gc_deg], 0.0)
            print("-- SETTING TARGET --")
            print(f"Target (ENU) x: {target_enu[0]:.3f} y: {target_enu[1]:.3f} z: {target_enu[2]:.3f}")
            print(f"Target Airspeed: {target_airspeed_kph:.3f}")

        self.sim[prp.target_ecef_x_m] = target_ecef_x_m
        self.sim[prp.target_ecef_y_m] = target_ecef_y_m
        self.sim[prp.target_ecef_z_m] = target_ecef_z_m
        self.sim[prp.target_airspeed_kph] = target_airspeed_kph

        self.prev_target_x = target_ecef_x_m
        self.prev_target_y = target_ecef_y_m
        self.prev_target_z = target_ecef_z_m
        self.prev_target_airspeed = target_airspeed_kph


    def reset_target_state(self):
        print("--- RESETTING TARGET ---")
        init_target = np.array([self.sim[prp.ecef_x_m], self.sim[prp.ecef_y_m], self.sim[prp.ecef_z_m], self.sim[prp.airspeed_kph]])
        self.set_target_state(init_target)
        print("------------------------")


    def update_errors(self, first_err=False):
        # update error jsbsim properties
        self.sim[prp.ecef_x_err_m] = self.sim[prp.target_ecef_x_m] - self.sim[prp.ecef_x_m]
        self.sim[prp.ecef_y_err_m] = self.sim[prp.target_ecef_y_m] - self.sim[prp.ecef_y_m]
        self.sim[prp.ecef_z_err_m] = self.sim[prp.target_ecef_z_m] - self.sim[prp.ecef_z_m]
        self.sim[prp.airspeed_err_kph] = self.sim[prp.target_airspeed_kph] - self.sim[prp.airspeed_kph]


    # Progress based reward + Airspeed hold
    def get_reward(self, action: np.ndarray) -> float:
        r_w: dict = self.task_cfg.reward.weights
        r_progress = super().get_reward(action)  # r_progress [-1.5, ~1.5]

        # r_airspeed [-max_r, 0]
        r_airspeed = 0.0
        if r_w["r_airspeed"]["enabled"]:
            r_airspeed = -(r_w["r_airspeed"]["max_r"] * np.tanh(r_w["r_airspeed"]["tanh_scale"] * np.abs(self.sim[prp.airspeed_err_kph])))
            self.sim[prp.reward_airspeed] = r_airspeed

        r_total = r_progress + r_airspeed
        self.sim[prp.reward_total] = r_total

        return r_total


    # def get_reward(self, action: np.ndarray) -> float:
    #     r_w: dict = self.task_cfg.reward.weights

    #     r_dist = r_w["r_dist"]["max_r"] * np.tanh(r_w["r_dist"]["tanh_scale"] * self.dist_to_target)
    #     self.sim[prp.reward_dist] = r_dist

    #     r_airspeed = r_w["r_airspeed"]["max_r"] * np.tanh(r_w["r_airspeed"]["tanh_scale"] * np.abs(self.sim[prp.airspeed_err_kph]))
    #     self.sim[prp.reward_airspeed] = r_airspeed

    #     r_actvar = np.nan
    #     if r_w["act_var"]["enabled"]:
    #         mean_actvar = np.mean(np.abs(action - np.array(self.action_hist)[-2])) # normalized by distance between action limits
    #         r_actvar = r_w["act_var"]["max_r"] * np.tanh(r_w["act_var"]["tanh_scale"] * mean_actvar)
    #     self.sim[prp.reward_actvar] = r_actvar

    #     r_reached = 0.0
    #     # if self.is_waypoint_reached():
    #     #     r_reached = 100
    #     # self.sim[prp.reward_reached] = r_reached

    #     r_total = -(r_dist + r_airspeed + r_actvar) + r_reached
    #     self.sim[prp.reward_total] = r_total

    #     return r_total


class WaypointTrackingNoVa(WaypointTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z).
        The agent controls only the aileron and elevator, while the throttle is controlled by a PI controller
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

        if self.jsbsim_cfg.debug and WaypointTrackingNoVa.__name__ == self.__class__.__name__:
            self.print_MDP_info()

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
        self.pid_airspeed.set_reference(60)


class CourseAltTracking(WaypointTrackingENU):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky.
        The agent is provided with the waypoint position coordinates in ENU and the target altitude.
        It then computes the waypoint course angle, and tracks the waypoint using the angle and the given altitude.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.course_err_rad, # course error
            prp.altitude_err_m, # altitude error
            prp.airspeed_kph, # airspeed
            prp.course_rad,
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            # prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
            # prp.aileron_combined_pos_norm, prp.elevator_pos_norm, prp.throttle_pos # actuator positions
        )

        if self.task_cfg.mdp.act_pos_in_state == True:
            self.state_prps += (
                prp.aileron_combined_pos_norm, prp.elevator_pos_norm, prp.throttle_pos # actuator positions
            )
        else:
            self.state_prps += (
                prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
            )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps = (
            prp.wp_course_rad, # target course
            prp.uav_to_wp_course_rad, # line of sight course
            prp.target_altitude_m, # target altitude (asl)
            prp.target_airspeed_kph, # target airspeed
        )

        # ENU target position, for telemetry
        self.target_enu_prps = (
            prp.target_enu_e_m, 
            prp.target_enu_n_m,
            prp.target_enu_u_m,
        )

        self.error_prps = (
            prp.course_err_rad,
            prp.altitude_err_m,
            prp.airspeed_err_kph,
        )

        self.reward_prps = (
            prp.reward_total, 
            prp.reward_course, prp.reward_altitude,
            prp.reward_airspeed,
            prp.dist_to_target_m,
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.enu_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps + (prp.course_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and CourseAltTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def observe_state(self, first_obs=False):
        self.sim[prp.course_rad] = np.arctan2(
            self.sim[prp.v_east_fps],
            self.sim[prp.v_north_fps]
        )
        return super().observe_state(first_obs)


    def reset_ext_state_props(self):
        """
            Resets the initial course angle of the aircraft.
            Need to recompute it from the ground velocity vector, since JSBSim does not provide it.
        """
        # reset the ENU position of the aircraft
        super().reset_ext_state_props()
        # reset the course angle of the aircraft
        self.sim[prp.course_rad] = np.arctan2(
            self.sim[prp.v_east_fps],
            self.sim[prp.v_north_fps]
        )


    def set_target_state(self, target: np.ndarray) -> None:
        """
            Set the target state.
            Takes the waypoint coordinates in ENU and converts them to path tracking parameters.
        """
        target_enu_e_m, target_enu_n_m, target_enu_u_m = target
        if np.any(target != [self.prev_target_x, self.prev_target_y, self.prev_target_z]):
            print("\t-- SETTING TARGET --")
            # Convert waypoint coordinates to path tracking parameters
            wp_path_course_rad, wp_path_alt_m = conversions.wpENU_to_wpCourseAlt(target).flatten()
            print(f"\tTarget (ENU) x: {target_enu_e_m:.3f} y: {target_enu_n_m:.3f} z: {target_enu_u_m:.3f}")
            print(f"\tCourse (Init/Target): {wp_path_course_rad:.3f} Altitude: {wp_path_alt_m:.3f}")

            self.sim[prp.wp_course_rad] = wp_path_course_rad
            self.sim[prp.target_altitude_m] = wp_path_alt_m
            self.sim[prp.target_airspeed_kph] = 60.0 # hardcoded target airspeed
            self.sim[prp.target_enu_e_m] = target_enu_e_m
            self.sim[prp.target_enu_n_m] = target_enu_n_m
            self.sim[prp.target_enu_u_m] = target_enu_u_m

        self.prev_target_x = target_enu_e_m
        self.prev_target_y = target_enu_n_m
        self.prev_target_z = target_enu_u_m


    def reset_target_state(self):
        """
            Resets the target state to the current state
        """
        print("--- RESETTING TARGET ---")
        init_target = np.array([0, 0, 600])
        self.set_target_state(init_target)
        print("------------------------")


    def update_errors(self, first_err=False):
        """
            Updates the errors based on the current state.
        """

        # self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad]
        # if (self.sim[prp.wp_course_rad] - self.sim[prp.course_rad]) < -np.pi:
        #     self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad] + 2*np.pi
        # elif (self.sim[prp.wp_course_rad] - self.sim[prp.course_rad]) > np.pi:
        #     self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad] - 2*np.pi
        # self.sim[prp.course_err_rad] = self.sim[prp.wp_course_wrapped_rad] - self.sim[prp.course_rad]

        # Computing the course error : the line of sight to the waypoint
        uav_to_wp_n = self.sim[prp.target_enu_n_m] - self.sim[prp.enu_n_m]
        uav_to_wp_e = self.sim[prp.target_enu_e_m] - self.sim[prp.enu_e_m]
        self.sim[prp.uav_to_wp_course_rad] = np.arctan2(uav_to_wp_e, uav_to_wp_n)
        if (self.sim[prp.uav_to_wp_course_rad] - self.sim[prp.course_rad]) < -np.pi:
            self.sim[prp.uav_to_wp_course_rad] = self.sim[prp.uav_to_wp_course_rad] + 2*np.pi
        elif (self.sim[prp.uav_to_wp_course_rad] - self.sim[prp.course_rad]) > np.pi:
            self.sim[prp.uav_to_wp_course_rad] = self.sim[prp.uav_to_wp_course_rad] - 2*np.pi
        self.sim[prp.course_err_rad] = self.sim[prp.uav_to_wp_course_rad] - self.sim[prp.course_rad]

        self.sim[prp.altitude_err_m] = self.sim[prp.target_altitude_m] - self.sim[prp.altitude_sl_m]

        self.sim[prp.airspeed_err_kph] = self.sim[prp.target_airspeed_kph] - self.sim[prp.airspeed_kph]

        super().update_errors(first_err)


    def get_reward(self, action: np.ndarray) -> float:
        """
            Computes the reward based on the current state and action.
            The reward is a combination of the course error and altitude error.
        """
        assert self.task_cfg.reward.name == "wp_course_alt" or self.task_cfg.reward.name == "wp_course_alt_va"
        r_w: dict = self.task_cfg.reward.weights

        r_course = r_w["r_course"]["weight"] * np.clip(
            np.abs(self.sim[prp.course_err_rad]) / r_w["r_course"]["scale"],
            a_min=0.0,
            a_max=r_w["r_course"]["clip_max"]
        )
        self.sim[prp.reward_course] = r_course

        # altitude error
        # r_altitude = r_w["r_altitude"]["scale"] * np.tanh(r_w["r_altitude"]["tanh_scale"] * np.abs(self.sim[prp.altitude_err_m]))
        # [min, max] target alt is [-30, 30]m, range is 60m
        r_altitude = r_w["r_alt"]["weight"] * np.clip(
            np.abs(self.sim[prp.altitude_err_m]) / r_w["r_alt"]["scale"],
            a_min=0.0,
            a_max=r_w["r_alt"]["clip_max"]
        )
        self.sim[prp.reward_altitude] = r_altitude

        # total reward
        r_total = -(r_course + r_altitude)
        self.sim[prp.reward_total] = r_total

        return r_total
    

class CourseAltVaTracking(CourseAltTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky.
        The agent is provided with the waypoint position coordinates in ENU and the target altitude.
        It then computes the waypoint course angle, and tracks the waypoint using the angle and the given altitude.
        In addition the agent has to maintain an airspeed of 60 kph.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.course_err_rad, # course error
            prp.altitude_err_m, # altitude error
            prp.airspeed_err_kph, # airspeed error
            prp.airspeed_kph, # airspeed
            prp.course_rad, # course angle
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps + (prp.course_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and CourseAltVaTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def get_reward(self, action: np.ndarray) -> float:
        """
            Computes the reward based on the current state and action.
            The reward is a combination of the course error and altitude error.
        """
        assert self.task_cfg.reward.name == "wp_course_alt_va"
        r_w: dict = self.task_cfg.reward.weights
        # get course and altitude rewards from the parent class
        r_course_alt = super().get_reward(action)

        # airspeed error
        r_airspeed = r_w["r_airspeed"]["weight"] * np.clip(
            np.abs(self.sim[prp.airspeed_err_kph]) / r_w["r_airspeed"]["scale"],
            a_min=0.0,
            a_max=r_w["r_airspeed"]["clip_max"]
        )
        self.sim[prp.reward_airspeed] = r_airspeed

        # total reward
        r_total = r_course_alt - r_airspeed
        self.sim[prp.reward_total] = r_total

        return r_total


class CourseAltNoVaTracking(CourseAltTracking):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky (described by its
        x, y, z).
        The airspeed is maintained at 60 kph by a PI controller.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.course_err_rad, # course error
            prp.altitude_err_m, # altitude error
            prp.airspeed_kph, # airspeed
            prp.course_rad,
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
            # prp.aileron_combined_pos_norm, prp.elevator_pos_norm, prp.throttle_pos # actuator positions
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd
        )

        self.target_prps = (
            prp.wp_course_rad, # target course
            prp.uav_to_wp_course_rad, # line of sight course
            prp.target_altitude_m, # target altitude (asl)
            prp.target_airspeed_kph, # target airspeed
        )

        # ENU target position, for telemetry
        self.target_enu_prps = (
            prp.target_enu_e_m, 
            prp.target_enu_n_m,
            prp.target_enu_u_m,
        )

        self.error_prps = (
            prp.course_err_rad,
            prp.altitude_err_m,
            prp.airspeed_err_kph,
        )

        self.reward_prps = (
            prp.reward_total, 
            prp.reward_course, prp.reward_altitude,
            prp.reward_airspeed,
            prp.dist_to_target_m,
        )

        self.enu_prps = (
            prp.enu_e_m, prp.enu_n_m, prp.enu_u_m, # position in ENU
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.enu_prps + self.target_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps + (prp.course_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # PI controller for airspeed
        self.pid_airspeed = PID(kp=0.5, ki=0.1, kd=0.0,
                           dt=self.fdm_dt, trim=TrimPoint(), 
                           limit=AeroModel().throttle_limit, is_throttle=True
        )

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and CourseAltNoVaTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

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
        self.pid_airspeed.set_reference(60)


class StraightPathTracking(WaypointTrackingENU):
    """
        Waypoint Tracking task. The agent has to track a given waypoint in the sky.
        The agent is provided with the waypoint course angle (in radians) to follow and a target airspeed (in kph).
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps = (
            prp.course_err_rad, # course error
            prp.altitude_err_m, # altitude error
            prp.airspeed_kph, # airspeed
            prp.course_rad,
            prp.u_fps, prp.v_fps, prp.w_fps, # velocity
            prp.att_qx, prp.att_qy, prp.att_qz, prp.att_qw, # attitude quaternion
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.alpha_rad, prp.beta_rad, # angle of attack, sideslip
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd # last action
        )

        self.action_prps = (
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd
        )

        self.target_prps = (
            prp.wp_course_rad, # target waypoint course
            prp.uav_to_line_course_des_rad, # desired course towards the line path
            prp.target_altitude_m, # target altitude (asl)
        )

        # ENU target position, for telemetry
        self.target_enu_prps = (
            prp.target_enu_e_m, 
            prp.target_enu_n_m,
            prp.target_enu_u_m,
        )

        self.error_prps = (
            prp.course_err_rad,
            prp.altitude_err_m,
        )

        self.reward_prps = (
            prp.reward_total, 
            prp.reward_course, prp.reward_altitude,
            prp.dist_to_target_m,
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.target_enu_prps \
                            + self.error_prps + self.reward_prps + (prp.course_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.in_missed_sphere = False
        self.inout_missed_sphere = False
        self.target_reached = False

        if self.jsbsim_cfg.debug and StraightPathTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def observe_state(self, first_obs=False):
        self.sim[prp.course_rad] = np.arctan2(
            self.sim[prp.v_east_fps],
            self.sim[prp.v_north_fps]
        )
        return super().observe_state(first_obs)


    def reset_ext_state_props(self):
        """
            Resets the initial course angle of the aircraft.
            Need to recompute it from the ground velocity vector, since JSBSim does not provide it.
        """
        # reset the ENU position of the aircraft
        super().reset_ext_state_props()
        # reset the course angle of the aircraft
        self.sim[prp.course_rad] = np.arctan2(
            self.sim[prp.v_east_fps],
            self.sim[prp.v_north_fps]
        )


    def set_target_state(self, target: np.ndarray) -> None:
        """
            Set the target state.
            Takes the waypoint coordinates in ENU and converts them to path tracking parameters.
        """
        target_enu_e_m, target_enu_n_m, target_enu_u_m = target
        if np.any(target != [self.prev_target_x, self.prev_target_y, self.prev_target_z]):
            print("-- SETTING TARGET --")
            # Convert waypoint coordinates to path tracking parameters
            wp_path_course_rad, wp_path_alt_m = conversions.wpENU_to_wpCourseAlt(target).flatten()
            print(f"Target (ENU) x: {target_enu_e_m:.3f} y: {target_enu_n_m:.3f} z: {target_enu_u_m:.3f}")
            print(f"Target Course: {wp_path_course_rad:.3f} Altitude: {wp_path_alt_m:.3f}")

            self.sim[prp.wp_course_rad] = wp_path_course_rad
            self.sim[prp.target_altitude_m] = wp_path_alt_m
            self.sim[prp.target_enu_e_m] = target_enu_e_m
            self.sim[prp.target_enu_n_m] = target_enu_n_m
            self.sim[prp.target_enu_u_m] = target_enu_u_m

        self.prev_target_x = target_enu_e_m
        self.prev_target_y = target_enu_n_m
        self.prev_target_z = target_enu_u_m


    def reset_target_state(self):
        """
            Resets the target state to the current state
        """
        print("--- RESETTING TARGET ---")
        init_target = np.array([0, 0, 600])
        self.set_target_state(init_target)
        print("------------------------")


    def update_errors(self, first_err=False):
        """
            Updates the errors based on the current state.
        """

        self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad]
        if (self.sim[prp.wp_course_rad] - self.sim[prp.course_rad]) <= -np.pi:
            self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad] + 2*np.pi
        elif (self.sim[prp.wp_course_rad] - self.sim[prp.course_rad]) > np.pi:
            self.sim[prp.wp_course_wrapped_rad] = self.sim[prp.wp_course_rad] - 2*np.pi

        prev_wp_enu_e = 0.0
        prev_wp_enu_n = 0.0
        path_params = self.task_cfg.reward.line_path_params
        cross_track_err = -np.sin(self.sim[prp.wp_course_wrapped_rad]) * (self.sim[prp.enu_n_m] - prev_wp_enu_n) + \
            np.cos(self.sim[prp.wp_course_wrapped_rad]) * (self.sim[prp.enu_e_m] - prev_wp_enu_e)
        self.sim[prp.uav_to_line_course_des_rad] = self.sim[prp.wp_course_wrapped_rad] - path_params["course_inf"] * \
            2/np.pi * np.arctan(path_params["k_path"] * cross_track_err)
        self.sim[prp.course_err_rad] = self.sim[prp.uav_to_line_course_des_rad] - self.sim[prp.course_rad]

        self.sim[prp.altitude_err_m] = self.sim[prp.target_altitude_m] - self.sim[prp.altitude_sl_m]

        super().update_errors(first_err)


    def get_reward(self, action: np.ndarray) -> float:
        """
            Computes the reward based on the current state and action.
            The reward is a combination of the course error and altitude error.
        """
        assert self.task_cfg.reward.name == "wp_st_path"
        r_w: dict = self.task_cfg.reward.weights

        r_course = r_w["r_course"]["weight"] * np.clip(
            np.abs(self.sim[prp.course_err_rad]) / r_w["r_course"]["scale"],
            a_min=0.0,
            a_max=r_w["r_course"]["clip_max"]
        )
        self.sim[prp.reward_course] = r_course

        # altitude error
        # r_altitude = r_w["r_altitude"]["scale"] * np.tanh(r_w["r_altitude"]["tanh_scale"] * np.abs(self.sim[prp.altitude_err_m]))
        # [min, max] target alt is [-30, 30]m, range is 60m
        r_altitude = r_w["r_alt"]["weight"] * np.clip(
            np.abs(self.sim[prp.altitude_err_m]) / r_w["r_alt"]["scale"],
            a_min=0.0,
            a_max=r_w["r_alt"]["clip_max"]
        )
        self.sim[prp.reward_altitude] = r_altitude

        # total reward
        r_total = -(r_course + r_altitude)
        self.sim[prp.reward_total] = r_total

        return r_total


class AltitudeTracking(JSBSimTask):
    def __init__(self, cfg_env, telemetry_file: str='', render_mode: str='none'):
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps: Tuple[BoundedProperty] = (
            prp.ecef_z_err_m,
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
            prp.target_ecef_z_m,
        )

        self.error_prps: Tuple[BoundedProperty] = (
            prp.ecef_z_err_m,
        )

        self.reward_prps: Tuple[BoundedProperty] = (
            prp.reward_total,
        )

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.prev_target_z = 0.0

        if self.jsbsim_cfg.debug and AltitudeTracking.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def reset_target_state(self):
        self.set_target_state(np.array([self.sim[prp.ecef_z_m]]))


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
        self.sim[prp.target_ecef_z_m] = target_state[0]
        self.prev_target_z = target_state[0]


    def update_errors(self, first_err=False):
        self.sim[prp.ecef_z_err_m] = self.sim[prp.target_ecef_z_m] - self.sim[prp.ecef_z_m]


    def apply_action(self, action):
        super().apply_action(action)
        self.sim[prp.aileron_cmd] = TrimPoint().aileron


    def get_reward(self, action):
        r_dist = np.abs(self.sim[prp.ecef_z_err_m])
        r_dist = 10 * np.tanh(0.005 * r_dist)
        r_total = -r_dist
        self.sim[prp.reward_total] = r_total
        return r_total

