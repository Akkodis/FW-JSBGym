import numpy as np
from typing import Tuple

from jsbgym.envs.tasks.attitude_control.ac_bohn_nova import ACBohnNoVaTask, ACBohnNoVaIErrTask
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty

class ACVanillaTask(ACBohnNoVaTask):
    """
        Attitude control task with minimal state and action space.
        No integral error, no wind states (alpha, beta), no past action nor action average.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
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

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


class ACVanillaActTask(ACVanillaTask):
    """
        Same as the parent class, but addition of previous action in the current state.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.aileron_cmd, prp.elevator_cmd
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


class ACVanillaAlphaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of the angle of attack in the state space.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.alpha_rad
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


class ACVanillaBetaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of the sideslip angle in the state space.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.beta_rad
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


class ACVanillaAlphaBetaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of both AoA and sideslip angles in the state space.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.alpha_rad, prp.beta_rad
        )

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


class ACVanillaIErrTask(ACBohnNoVaIErrTask):
    """
        Attitude control task with vanilla state and action space.
        But with integral error states.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err, # integral errors
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
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)

