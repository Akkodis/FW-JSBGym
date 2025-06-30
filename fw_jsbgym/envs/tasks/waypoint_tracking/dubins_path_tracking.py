import numpy as np

from omegaconf import DictConfig
from typing import Tuple, Dict
from fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking import WaypointTrackingENU, CourseAltTracking
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.conversions import angle_rad_wrap_to_pi
from dubins_path_planning.trajectory_planning import DubinsManeuver3D_func, compute_sampling


class DubinsPathTrackingv0(CourseAltTracking):
    """
        Dubins Path Tracking Environment - version 0, it doesn't change anything from the CourseAltTracking state/action space nor the reward.
        Just takes the intermediate Dubins paths and feeds them to a CourseAltTracking type task.
        This environment is designed for tracking Dubins paths, which are paths that consist of straight lines and circular arcs.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env=cfg_env, telemetry_file=telemetry_file, render_mode=render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        # self.state_prps = ()

        # self.action_prps = ()

        # self.target_prps = ()

        # self.target_enu_prps = ()

        # self.error_prps = ()

        # self.reward_prps = ()

        # self.telemetry_prps = ()

        self.action_space = self.get_action_space()

        self.observation_space = self.get_observation_space()

        self.dist_to_final_target: float = 0.0

        self.prev_final_target_x: float = 0.0
        self.prev_final_target_y: float = 0.0
        self.prev_final_target_z: float = 0.0

        self.dubins_points: np.ndarray = np.zeros((0, 5))  # Array to hold the Dubins path points
        self.dubins_pt_idx: int = 0

        if self.jsbsim_cfg.debug:
            self.print_MDP_info()
        
        self.telemetry_setup(self.telemetry_file)
    

    def reset_target_state(self):
        """
            Resets the target state to the current state
        """
        self.dubins_pt_idx = 0
        self.sim[prp.final_target_enu_e_m] = 0.0
        self.sim[prp.final_target_enu_n_m] = 0.0
        self.sim[prp.final_target_enu_u_m] = 600.0
        super().reset_target_state()


    def set_target_state(self, target: np.ndarray):
        """
            Sets the target state for the environment
            :param target: Final target state as a numpy array [x, y, z] (ENU coordinates)
        """
        # print("Setting target state...")
        if np.all(target == [0, 0, 600]): # if target is the initial state
            super().set_target_state(target)
            self.dubins_pt_idx = 0
            self.prev_final_target_x = target[0]
            self.prev_final_target_y = target[1]
            self.prev_final_target_z = target[2]
            return
        elif np.any(target != [self.prev_final_target_x, self.prev_final_target_y, self.prev_final_target_z]):
            print(f"New final target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            # compute intermediary Dubins path points
            self.dubins_points = self.compute_dubins_path(target)
            print(f"Last Dubins point: {self.dubins_points[-1]}")

        # print(f"Setting target state: {target}")

        # Check logic to change dubins current point to track
        current_pos_enu = np.array([self.sim[prp.enu_e_m], self.sim[prp.enu_n_m], self.sim[prp.enu_u_m]])
        current_dubins_pt = self.dubins_points[self.dubins_pt_idx]
        dist_to_curr_dubins_pt = np.sqrt(np.sum((current_pos_enu - current_dubins_pt[:3]) ** 2))
        if dist_to_curr_dubins_pt < self.task_cfg.mdp.dub_switch_dist:  # If within the switching distance of the current Dubins point, switch to the next point
            if self.dubins_pt_idx < len(self.dubins_points) - 1:
                self.dubins_pt_idx += 1

            self.sim[prp.final_target_enu_e_m] = target[0]
            self.sim[prp.final_target_enu_n_m] = target[1]
            self.sim[prp.final_target_enu_u_m] = target[2]
            print(f"Tracking Dubins point {self.dubins_pt_idx}/{len(self.dubins_points)}")
            super().set_target_state(self.dubins_points[self.dubins_pt_idx, :3])
            self.sim[prp.dubins_target_course_rad] = self.dubins_points[self.dubins_pt_idx, 3]  # Set the target course from the Dubins points
            self.sim[prp.dubins_target_flightpath_rad] = self.dubins_points[self.dubins_pt_idx, 4]  # Set the target flight path angle from the Dubins points
            print(f"\tTarget Course: {self.sim[prp.dubins_target_course_rad]:.3f} rad, Target Flight Path: {self.sim[prp.dubins_target_flightpath_rad]:.3f} rad")

        self.prev_final_target_x = target[0]
        self.prev_final_target_y = target[1]
        self.prev_final_target_z = target[2]


    def compute_dubins_path(self, final_target: np.ndarray):
        """
            Computes the Dubins path between the current state and the target state
            :return: List of points representing the Dubins path
        """
        print("Computing Dubins path...")

        # Initial and final configurations [x, y, z, heading angle, pitch angle]
        e_i, n_i, u_i = 0, 0, 600
        e_f, n_f, u_f = final_target[0], final_target[1], final_target[2] # swapped to match ENU coordinates

        # final heading angle is computed as the angle between the initial and final points
        # giving XYZ coordinates to the DubinsManeuver3D_func, will convert it to ENU heading later
        psi_i, psi_f = np.deg2rad(90.), np.arctan2(n_f - n_i, e_f - e_i)
        qi = np.array([e_i, n_i, u_i, psi_i, 0.0])
        qf = np.array([e_f, n_f, u_f, psi_f, 0.0])

        # Pitch angle constraints [min_pitch, max_pitch]
        pitch_max = np.array(self.task_cfg.mdp.pitch_max) * np.pi / 180.0
        maneuver = DubinsManeuver3D_func(qi, qf, self.task_cfg.mdp.rho_min, pitch_max)
        num_of_samples = max(int(np.floor(maneuver.length / self.task_cfg.mdp.dub_sampling_dist)), 2)

        # Sample the maneuver
        samples = compute_sampling(maneuver, numberOfSamples=num_of_samples)
        samples = np.array(samples)
        samples[:, 3] = angle_rad_wrap_to_pi(np.pi/2 - samples[:, 3]) # Convert to ENU-North-relative heading angle and [-pi, pi]
        samples[:, 4] = angle_rad_wrap_to_pi(samples[:, 4])  # Ensure flight path angles are wrapped to [-pi, pi]

        return samples


    def is_waypoint_reached(self):
        """
            Returns True if the distance to the target is less than 3 meters.
        """
        self.target_reached = False
        if self.dist_to_final_target < 3:
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
        if self.dist_to_final_target < 10.0:
            self.in_missed_sphere = True
        # UAV exits the sphere
        if self.in_missed_sphere and self.dist_to_final_target > 10.0:
            self.in_missed_sphere = False
            self.inout_missed_sphere = True
            print(f"Target Missed! @ step : {self.sim[self.current_step]}")
        return self.inout_missed_sphere
    

    def observe_state(self, first_obs = False):
        state = super().observe_state(first_obs)
        if first_obs:
            self.dist_to_final_target = 0.0
        else:
            self.dist_to_final_target = np.sqrt(
                self.sim[prp.enu_e_final_err_m] ** 2 +
                self.sim[prp.enu_n_final_err_m] ** 2 +
                self.sim[prp.enu_u_final_err_m] ** 2
            )
        self.sim[prp.dist_to_final_target_m] = self.dist_to_final_target
        return state

    
    def update_errors(self, first_err: bool=False):
        self.sim[prp.enu_e_final_err_m] = self.sim[prp.final_target_enu_e_m] - self.sim[prp.enu_e_m]
        self.sim[prp.enu_n_final_err_m] = self.sim[prp.final_target_enu_n_m] - self.sim[prp.enu_n_m]
        self.sim[prp.enu_u_final_err_m] = self.sim[prp.final_target_enu_u_m] - self.sim[prp.enu_u_m]
        super().update_errors(first_err)

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
            Step the environment with the given action.
            :param action: Action to take
            :return: Tuple of (next_state, reward, done, info)
        """
        next_state, reward, terminated, truncated, info = super().step(action)

        # add the dubins point trajectory to the info dictionary
        info["dubins_points"] = self.dubins_points

        return next_state, reward, terminated, truncated, info


class DubinsPathTrackingv1(DubinsPathTrackingv0):
    """
        Dubins Path Tracking Environment
        This environment is designed for tracking Dubins paths, which are paths that consist of straight lines and circular arcs.
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
            prp.aileron_combined_pos_norm, prp.elevator_pos_norm, prp.throttle_pos, # actuator positions
            prp.dubins_target_course_err, prp.dubins_target_flightpath_err, # Dubins intermediate target errors (course and flight path angle errors)
        ) 

        # self.action_prps = ()

        self.target_prps += (
            prp.dubins_target_course_rad,  # Target course angle in radians
            prp.dubins_target_flightpath_rad,  # Target flight path angle in radians
        )

        self.target_enu_prps = ()

        # self.error_prps = ()

        # self.reward_prps = ()

        self.telemetry_prps += (
            prp.dubins_target_course_rad,
            prp.dubins_target_flightpath_rad,
            prp.dubins_target_course_err,
            prp.dubins_target_flightpath_err,
        )

        self.action_space = self.get_action_space()

        self.observation_space = self.get_observation_space()

        if self.jsbsim_cfg.debug:
            self.print_MDP_info()
        
        self.telemetry_setup(self.telemetry_file)


    def set_target_state(self, target: np.ndarray):
        """
            Sets the target state for the environment
            :param target: Final target state as a numpy array [x, y, z] (ENU coordinates)
        """
        super().set_target_state(target)
        if np.all(target == [0, 0, 600]):
            self.sim[prp.dubins_target_course_rad] = 0.0
            self.sim[prp.dubins_target_flightpath_rad] = 0.0


    def update_errors(self, first_err: bool=False):
        """
            Update the errors in the simulation state.
            :param first_err: If True, this is the first error update
        """
        self.sim[prp.dubins_target_course_err] = self.sim[prp.dubins_target_course_rad] - self.sim[prp.course_rad]
        self.sim[prp.dubins_target_flightpath_err] = self.sim[prp.dubins_target_flightpath_rad] - self.sim[prp.flightpath_gamma_rad]
        super().update_errors(first_err)
