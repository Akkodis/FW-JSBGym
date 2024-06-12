import numpy as np
from typing import Tuple
from omegaconf import DictConfig

from jsbgym.envs.tasks.attitude_control.ac_bohn import ACBohnTask
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.agents.pid import PID
from jsbgym.models.aerodynamics import AeroModel


class ACBohnNoVaTask(ACBohnTask):
    """
        gym.Env wrapper task. Made for attitude control without airspeed control. Airspeed is controlled by a PI controller.

        Attr:
            - `state_vars`: Tuple of BoundedProperty objects, defining the structure of an aircraft state (to be observed by the agent)
            - `action_vars`: Tuple of BoundedProperty objects, defining the structure of the action variables representing the control surface commands
            - `target_state_vars`: Tuple of BoundedProperty objects, defining the target state variables representing the reference setpoint for the controller to track
            - `telemetry_vars`: Tuple of BoundedProperty objects, defining the telemetry state variables representing the state of the aircraft to be logged
            - `telemetry_file`: the name of the file containing the flight data to be logged
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        """
            Args: 
                - `cfg_env`: DictConfig object containing the environment configuration
                - `telemetry_file`: the name of the file containing the flight data to be logged
                - `render_mode`: the render mode for the task
        """
        super().__init__(cfg_env, telemetry_file, render_mode)

    ### STATE SPACE ###
        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.alpha_rad, prp.beta_rad # angle of attack and sideslip angle
        )

        # if action_avg is enabled, use the average of the past N commands in the state space
        if self.task_cfg.mdp.get("action_avg", False):
            self.state_prps += (prp.aileron_avg, prp.elevator_avg)
        else: # otherwise, use the fcs positions variables
            self.state_prps += (prp.aileron_cmd, prp.elevator_cmd)
            # self.state_prps += (prp.aileron_combined_pos_rad, prp.elevator_pos_rad) # fcs positions
            # self.state_prps += (prp.aileron_combined_pos_norm, prp.elevator_pos_norm) # fcs positions normalized
    ### STATE SPACE ###

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.aileron_cmd, prp.elevator_cmd
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            prp.target_roll_rad, prp.target_pitch_rad # target attitude
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, # errors
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # PI controller for airspeed
        self.pid_airspeed = PID(kp=0.5, ki=0.1, kd=0.0,
                           dt=self.fdm_dt, trim=TrimPoint(), 
                           limit=AeroModel().throttle_limit, is_throttle=True
        )
        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)

        self.prev_target_roll: float = 0.0
        self.prev_target_pitch: float = 0.0


    def apply_action(self, action: np.ndarray) -> None:
        """
            Apply the action to the simulation + maintain airspeed at 60 kph with PI controller.
        """
        for prop, command in zip(self.action_prps, action):
            self.sim[prop] = command
        # self.sim[prp.throttle_cmd] = TrimPoint().throttle # set throttle to trim point throttle
        # maintain airspeed at 60 kph with PI controller
        self.pid_airspeed.set_reference(60)
        throttle_cmd, airspeed_err, _ = self.pid_airspeed.update(state=self.sim[prp.airspeed_kph], saturate=True)
        self.sim[prp.throttle_cmd] = throttle_cmd


    def update_errors(self) -> None:
        """
            Update the error properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties

        self.sim[prp.roll_err] = self.sim[prp.target_roll_rad] - self.sim[prp.roll_rad]
        self.sim[prp.pitch_err] = self.sim[prp.target_pitch_rad] - self.sim[prp.pitch_rad]

        # fill errors namedtuple with error variable values from the sim properties
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_prps])


    def update_action_avg(self) -> None:
        """
            Update the average of the past N commands (elevator, aileron, throttle)
        """
        self.sim[prp.aileron_avg] = np.mean(np.array(self.action_hist)[:, 0])
        self.sim[prp.elevator_avg] = np.mean(np.array(self.action_hist)[:, 1])


    def set_target_state(self, target_roll_rad: float, target_pitch_rad: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
        """
        if target_roll_rad != self.prev_target_roll:
            print("Target roll: ", np.rad2deg(target_roll_rad))
        if target_pitch_rad != self.prev_target_pitch:
            print("Target pitch: ", np.rad2deg(target_pitch_rad))

        # set target state sim properties
        self.sim[prp.target_roll_rad] = target_roll_rad
        self.sim[prp.target_pitch_rad] = target_pitch_rad

        self.prev_target_roll = target_roll_rad
        self.prev_target_pitch = target_pitch_rad

        # fill target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_roll_rad), str(target_pitch_rad))


    def reset_target_state(self) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset airspeed pid integral error
        self.pid_airspeed.reset()

        # reset task class attributes with initial conditions
        self.set_target_state(target_roll_rad=self.sim[prp.initial_roll_rad], 
                              target_pitch_rad=self.sim[prp.initial_pitch_rad])


    def get_reward_bohnorig(self, action: np.ndarray) -> float:
        """
            Reward function
            Based on the bohn PPO paper reward func no airspeed control.
            Action penalty is a moving average of the differences between consecutives actions over the past N actions.
        """
        r_w: dict = self.task_cfg.reward.weights # reward weights for each reward component
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], r_w["roll"]["clip_min"], r_w["roll"].get("clip_max", None)) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], r_w["pitch"]["clip_min"], r_w["pitch"].get("clip_max", None)) # pitch reward component
        r_airspeed = np.clip(abs(self.sim[prp.airspeed_err]) / r_w["Va"]["scaling"], r_w["Va"]["clip_min"], r_w["Va"].get("clip_max", None)) # airspeed reward component

        # r_act_low: np.ndarray = np.where(action < self.action_space.low, self.action_space.low - action, 0)
        # r_act_high: np.ndarray = np.where(action > self.action_space.high, action - self.action_space.high, 0)
        # r_act_bounds_raw: float = np.sum(np.abs(r_act_low) + np.sum(np.abs(r_act_high))) # doute sur le np.sum
        # r_act_bounds: float = np.clip(r_act_bounds_raw / r_w["act_bounds"]["scaling"], 0, r_w["act_bounds"].get("clip_max", None))

        np_action_hist: np.ndarray = np.array(self.action_hist)
        deltas: np.ndarray = np.diff(np_action_hist[-self.task_cfg.mdp.obs_hist_size:], axis=0)
        r_actvar_raw = np.sum(np.abs(deltas))
        r_actvar = np.clip(r_actvar_raw / r_w["act_var"]["scaling"], r_w["act_var"]["clip_min"], r_w["act_var"].get("clip_max", None)) # flight control surface reward component

        # return the negative sum of all reward components
        # r_total: float = -(r_roll + r_pitch + r_airspeed + r_actvar + r_act_bounds)
        r_total: float = -(r_roll + r_pitch + r_airspeed + r_actvar) # removed action bound penalty since it's not used in the paper but only in the code.

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_airspeed] = r_airspeed
        self.sim[prp.reward_actvar] = r_actvar
        # self.sim[prp.reward_act_bounds] = r_act_bounds
        self.sim[prp.reward_total] = r_total

        return r_total


    def get_reward(self, action: np.ndarray) -> float:
        """
            Reward function
            Based on the bohn PPO paper reward func no airspeed control.
            Choice between bohn_mod (modified, action penalty between consecutive actions)
            and bohn_orig reward components (orig, from the paper, sum of absolute differences between N consecutive actions)
        """
        r_w: dict = self.task_cfg.reward.weights # reward weights for each reward component
        r_roll_clip_max = r_w["roll"].get("clip_max", None)
        r_pitch_clip_max = r_w["pitch"].get("clip_max", None)

        # roll and pitch error reward (penalty) components
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], 0.0, r_roll_clip_max) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], 0.0, r_pitch_clip_max) # pitch reward component

        r_actvar = np.nan
        r_actvar_raw = np.nan
        # action fluctuation (penalty) reward component
        if r_w["act_var"]["enabled"]:
            r_act_clip_max = r_w["act_var"].get("clip_max", None)
            if r_roll_clip_max + r_pitch_clip_max + r_act_clip_max != 1.0:
                print(f"WARNING: Reward components do not sum to 1.0")
            # either use the bohnmod (diff between consective actions) reward component or the bohnorig reward component (sum of absolute differences between N consecutive actions)
            if self.task_cfg.reward.name == "bohn_mod":
                r_actvar = np.mean(np.abs(action - np.array(self.action_hist)[-2])) / 2*self.action_space.high[0] # normalized by distance between min and max action value dist(-1, 1)=2
                r_actvar = np.clip(r_actvar, 0.0, r_act_clip_max)
            elif self.task_cfg.reward.name == "bohn_orig":
                np_action_hist: np.ndarray = np.array(self.action_hist)
                deltas: np.ndarray = np.diff(np_action_hist[-self.task_cfg.mdp.act_hist_size:], axis=0)
                r_actvar_raw = np.sum(np.abs(deltas))
                r_actvar = np.clip(r_actvar_raw / r_w["act_var"]["scaling"], 0.0, r_w["act_var"].get("clip_max", None)) # flight control surface reward component
            r_total: float = -(r_roll + r_pitch + r_actvar)
        else:
            if r_roll_clip_max + r_pitch_clip_max != 1.0:
                print("WARNING: Reward components do not sum to 1.0")
            r_total: float = -(r_roll + r_pitch)

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_actvar] = r_actvar
        self.sim[prp.reward_actvar_raw] = r_actvar_raw # not actually returned as a reward component but useful for debugging
        self.sim[prp.reward_total] = r_total

        return r_total


class ACBohnNoVaIErrTask(ACBohnNoVaTask):
    """
        Same as the parent class.
        Added integral errors to the state variables and re-implemented some methods to update the integral errors.
        Added angle of attack and sideslip angle to the state variables.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.state_prps += (prp.roll_integ_err, prp.pitch_integ_err) # integral errors

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err # integral errors
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def set_target_state(self, target_roll_rad: float, target_pitch_rad: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
            If the target state changes, reset the integral errors.
        """
        # if there's a change in target state, reset integral errors
        if target_roll_rad != self.prev_target_roll:
            self.sim[prp.roll_integ_err] = 0.0
            print("Target roll: ", np.rad2deg(target_roll_rad))
        if target_pitch_rad != self.prev_target_pitch:
            self.sim[prp.pitch_integ_err] = 0.0
            print("Target pitch: ", np.rad2deg(target_pitch_rad))

        self.sim[prp.target_roll_rad] = target_roll_rad
        self.sim[prp.target_pitch_rad] = target_pitch_rad
        self.prev_target_roll = target_roll_rad
        self.prev_target_pitch = target_pitch_rad

        # fill target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_roll_rad), str(target_pitch_rad))


    def update_errors(self) -> None:
        """
            Update the error and integral errors properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties
        self.sim[prp.roll_err] = self.sim[prp.target_roll_rad] - self.sim[prp.roll_rad]
        self.sim[prp.pitch_err] = self.sim[prp.target_pitch_rad] - self.sim[prp.pitch_rad]
        self.sim[prp.roll_integ_err] = 1.00 * self.sim[prp.roll_integ_err] + self.sim[prp.roll_err] * 0.01
        self.sim[prp.pitch_integ_err] = 1.00 * self.sim[prp.pitch_integ_err] + self.sim[prp.pitch_err] * 0.01
        # print(f"roll integ err: {self.sim[prp.roll_integ_err]}")
        # print(f"pitch integ err: {self.sim[prp.pitch_integ_err]}")

        # fill errors namedtuple with error variable values from the sim properties
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_prps])


    def reset_target_state(self) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset task class attributes with initial conditions (use the parent class method)
        super().reset_target_state()
        print("resetting agent integral errors")
        # reset integral errors
        self.sim[prp.roll_integ_err] = 0.0
        self.sim[prp.pitch_integ_err] = 0.0


class ACBohnNoVaIErrYawTask(ACBohnNoVaIErrTask):
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.state_prps += (prp.heading_rad, ) # integral errors

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)
