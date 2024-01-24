import numpy as np
from typing import Tuple

from jsbgym.envs.tasks.attitude_control import AttitudeControlTask
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty
from jsbgym.trim.trim_point import TrimPoint
from jsbgym.agents.pid import PID
from jsbgym.models.aerodynamics import AeroModel


class ACNoVaTask(AttitudeControlTask):
    """
        gym.Env wrapper task. Made for attitude control without airspeed control.

        Attr:
            - `state_vars`: Tuple of BoundedProperty objects, defining the structure of an aircraft state (to be observed by the agent)
            - `action_vars`: Tuple of BoundedProperty objects, defining the structure of the action variables representing the control surface commands
            - `target_state_vars`: Tuple of BoundedProperty objects, defining the target state variables representing the reference setpoint for the controller to track
            - `telemetry_vars`: Tuple of BoundedProperty objects, defining the telemetry state variables representing the state of the aircraft to be logged
            - `telemetry_file`: the name of the file containing the flight data to be logged
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        """
            Args: 
                - `config_file`: the name of the config file containing the task parameters
                - `telemetry_file`: the name of the file containing the flight data to be logged
                - `render_mode`: the render mode for the task
        """
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.aileron_avg, prp.elevator_avg # average of past 5 fcs commands
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.aileron_cmd, prp.elevator_cmd
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            prp.target_roll_rad, prp.target_pitch_rad # target attitude
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, # errors
        )

        self.telemetry_prps: Tuple[BoundedProperty, ...] = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates and airspeed
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, # control surface commands
            prp.reward_total, prp.reward_roll, prp.reward_pitch, # rewards
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps + self.error_prps # target state variables

        self.pid_airspeed = PID(kp=0.5, ki=0.1, kd=0.0,
                           dt=self.fdm_dt, trim=TrimPoint(), # TODO: make setting of dt more modulable (read from config file)
                           limit=AeroModel().throttle_limit, is_throttle=True
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def apply_action(self, action: np.ndarray) -> None:
        # apply the action to the simulation
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
        # set target state sim properties
        self.sim[prp.target_roll_rad] = target_roll_rad
        self.sim[prp.target_pitch_rad] = target_pitch_rad

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


    def get_reward(self, action: np.ndarray) -> float:
        """
            Reward function
            Based on the bohn PPO paper reward func no airspeed control.
        """
        r_w: dict = self.task_cfg["reward_weights"] # reward weights for each reward component
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], r_w["roll"]["clip_min"], r_w["roll"].get("clip_max", None)) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], r_w["pitch"]["clip_min"], r_w["pitch"].get("clip_max", None)) # pitch reward component

        # return the negative sum of all reward components
        r_total: float = -(r_roll + r_pitch)

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_total] = r_total

        return r_total


class ACNoVaIntegErrTask(ACNoVaTask):
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
        prp.roll_rad, prp.pitch_rad, # attitude
        prp.airspeed_kph, # airspeed
        prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
        prp.roll_err, prp.pitch_err, # errors
        prp.roll_integ_err, prp.pitch_integ_err, # integral errors
        prp.aileron_avg, prp.elevator_avg, # average of past 5 fcs commands
        prp.alpha_rad, prp.beta_rad # angle of attack and sideslip angles
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err # integral errors
        )

        self.telemetry_prps: Tuple[BoundedProperty, ...] = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates and airspeed
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, # control surface commands
            prp.reward_total, prp.reward_roll, prp.reward_pitch, # rewards
            # prp.reward_int_roll, prp.reward_int_pitch, # integral rewards
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.prev_target_roll: float = 0.0
        self.prev_target_pitch: float = 0.0

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def set_target_state(self, target_roll_rad: float, target_pitch_rad: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
        """
        # if there's a change in target state, reset integral errors
        if target_roll_rad != self.prev_target_roll:
            self.sim[prp.roll_integ_err] = 0.0
            print("Target roll: ", target_roll_rad)
        if target_pitch_rad != self.prev_target_pitch:
            self.sim[prp.pitch_integ_err] = 0.0
            print("Target pitch: ", target_pitch_rad)

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


class ACNoVaPIDRLTask(ACNoVaTask):
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.kp_roll_act, prp.ki_roll_act, prp.kd_roll_act,
            prp.kp_pitch_act, prp.ki_pitch_act, prp.kd_pitch_act,
            prp.aileron_cmd, prp.elevator_cmd, # control surface commands (output of the PID controller)
            prp.alpha_rad, prp.beta_rad # angle of attack and sideslip angles
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.kp_roll, prp.ki_roll, prp.kd_roll,
            prp.kp_pitch, prp.ki_pitch, prp.kd_pitch
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            prp.target_roll_rad, prp.target_pitch_rad # target attitude
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, # errors
        )

        self.telemetry_prps: Tuple[BoundedProperty, ...] = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates and airspeed
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, # control surface commands
            prp.reward_total, prp.reward_roll, prp.reward_pitch, # rewards
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps + self.error_prps + self.action_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # PIDs and their initial gain values
        self.kp_roll_base: float = 1.5
        self.ki_roll_base: float = 0.1
        self.kd_roll_base: float = 0.1
        self.pid_roll = PID(kp=self.kp_roll_base, ki=self.ki_roll_base, kd=self.kd_roll_base,
                            dt=self.fdm_dt, 
                            limit=AeroModel().aileron_limit
                            # limit = 1.0
                            )

        self.kp_pitch_base: float = -2.0
        self.ki_pitch_base: float = -0.3
        self.kd_pitch_base: float = -0.1
        self.pid_pitch = PID(kp=self.kp_pitch_base, ki=self.ki_pitch_base, kd=self.kd_pitch_base,
                             dt=self.fdm_dt, 
                             limit=AeroModel().elevator_limit
                            #  limit = 1.0
                             )

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def reset_props(self, seed: int=None, options: dict=None) -> Tuple[np.ndarray, np.ndarray]:
        """
            Reset the task environment.
        """
        # populate the properties with the initial values
        super().reset_props()
        # reset the task actions i.e. the PID gains to their initial values
        print("resetting agent PID gains")
        self.sim[prp.kp_roll] = 0.0
        self.sim[prp.ki_roll] = 0.0
        self.sim[prp.kd_roll] = 0.0
        self.sim[prp.kp_pitch] = 0.0
        self.sim[prp.ki_pitch] = 0.0
        self.sim[prp.kd_pitch] = 0.0

        # reset the RL action additive terms of the PID gains to zero
        self.sim[prp.kp_roll_act] = 0.0
        self.sim[prp.ki_roll_act] = 0.0
        self.sim[prp.kd_roll_act] = 0.0
        self.sim[prp.kp_pitch_act] = 0.0
        self.sim[prp.ki_pitch_act] = 0.0
        self.sim[prp.kd_pitch_act] = 0.0


    def apply_action(self, action: np.ndarray) -> None:
        # doesn't have a direct effect in the simulation for roll and pitch
        # just sets the properties accordingly (useful for telemetry and logging)
        # and contrains the PI controller for throttle (maintain airspeed at 55 kph)
        super().apply_action(action)

        # apply the action (pitch and roll PID gains)
        self.sim[prp.kp_roll_act] = action[0]
        self.sim[prp.ki_roll_act] = action[1]
        self.sim[prp.kd_roll_act] = action[2]
        self.sim[prp.kp_roll] = self.kp_roll_base + self.sim[prp.kp_roll_act]
        self.sim[prp.ki_roll] = self.ki_roll_base + self.sim[prp.ki_roll_act]
        self.sim[prp.kd_roll] = self.kd_roll_base + self.sim[prp.kd_roll_act]
        self.pid_roll.set_gains(kp=self.sim[prp.kp_roll], ki=self.sim[prp.ki_roll], kd=self.sim[prp.kd_roll])

        self.sim[prp.kp_pitch_act] = action[3]
        self.sim[prp.ki_pitch_act] = action[4]
        self.sim[prp.kd_pitch_act] = action[5]
        self.sim[prp.kp_pitch] = self.kp_pitch_base + self.sim[prp.kp_pitch_act]
        self.sim[prp.ki_pitch] = self.ki_pitch_base + self.sim[prp.ki_pitch_act]
        self.sim[prp.kd_pitch] = self.kd_pitch_base + self.sim[prp.kd_pitch_act]
        self.pid_pitch.set_gains(kp=self.sim[prp.kp_pitch], ki=self.sim[prp.ki_pitch], kd=self.sim[prp.kd_pitch])

        aileron_cmd, _, _ = self.pid_roll.update(state=self.sim[prp.roll_rad], state_dot=self.sim[prp.p_radps], 
                                                 saturate=True, normalize=True)
        elevator_cmd, _, _ = self.pid_pitch.update(state=self.sim[prp.pitch_rad], state_dot=self.sim[prp.q_radps], 
                                                   saturate=True, normalize=True)

        self.sim[prp.aileron_cmd] = aileron_cmd
        self.sim[prp.elevator_cmd] = elevator_cmd


    def set_target_state(self, target_roll_rad: float, target_pitch_rad: float) -> None:
        # just sets the properties accordingly (useful for telemetry and logging) from the parent class
        super().set_target_state(target_roll_rad, target_pitch_rad)

        # set the targets for the PIDs
        self.pid_roll.set_reference(target_roll_rad)
        self.pid_pitch.set_reference(target_pitch_rad)


    def reset_target_state(self) -> None:
        super().reset_target_state()

        # reset all the PIDs
        self.pid_roll.reset()
        self.pid_pitch.reset()
