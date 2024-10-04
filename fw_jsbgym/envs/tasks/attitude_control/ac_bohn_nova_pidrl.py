import numpy as np
from typing import Tuple

from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.envs. tasks.attitude_control.ac_bohn_nova import ACBohnNoVaIErrTask
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty
from fw_jsbgym.agents.pid import PID
from fw_jsbgym.models.aerodynamics import AeroModel


class ACNoVaPIDRLAddTask(ACBohnNoVaIErrTask):
    """
        Attitude control task based on the Bohn state space definition with no airspeed control and where the
        PID gains are the actions to be learned by the RL agent. The PID gains are added to the base gains.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err, # integral errors
            prp.kp_roll_act, prp.ki_roll_act, prp.kd_roll_act,
            prp.kp_pitch_act, prp.ki_pitch_act, prp.kd_pitch_act,
            prp.aileron_cmd, prp.elevator_cmd, # control surface commands (output of the PID controller)
            prp.alpha_rad, prp.beta_rad # angle of attack and sideslip angles
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.kp_roll_act, prp.ki_roll_act, prp.kd_roll_act,
            prp.kp_pitch_act, prp.ki_pitch_act, prp.kd_pitch_act
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            prp.target_roll_rad, prp.target_pitch_rad # target attitude
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
            prp.kp_roll, prp.ki_roll, prp.kd_roll,
            prp.kp_pitch, prp.ki_pitch, prp.kd_pitch
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
        # self.sim[prp.kp_roll_act] = action[0]
        # self.sim[prp.ki_roll_act] = action[1]
        # self.sim[prp.kd_roll_act] = action[2]
        self.sim[prp.kp_roll] = self.kp_roll_base + self.sim[prp.kp_roll_act]
        self.sim[prp.ki_roll] = self.ki_roll_base + self.sim[prp.ki_roll_act]
        self.sim[prp.kd_roll] = self.kd_roll_base + self.sim[prp.kd_roll_act]
        self.pid_roll.set_gains(kp=self.sim[prp.kp_roll], ki=self.sim[prp.ki_roll], kd=self.sim[prp.kd_roll])

        # self.sim[prp.kp_pitch_act] = action[3]
        # self.sim[prp.ki_pitch_act] = action[4]
        # self.sim[prp.kd_pitch_act] = action[5]
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


class ACNoVaPIDRLTask(ACBohnNoVaIErrTask):
    """
        Attitude control task based on the Bohn state space definition with no airspeed control and where the
        PID gains are the actions to be learned by the RL agent.
    """
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(config_file, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err, # integral errors
            prp.kp_roll, prp.ki_roll, prp.kd_roll,
            prp.kp_pitch, prp.ki_pitch, prp.kd_pitch,
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
            prp.kp_roll, prp.ki_roll, prp.kd_roll,
            prp.kp_pitch, prp.ki_pitch, prp.kd_pitch
        ) + self.target_prps + self.error_prps + self.action_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # PIDs and their initial gain values
        self.pid_roll = PID(kp=0.0, ki=0.0, kd=0.0,
                            dt=self.fdm_dt, 
                            limit=AeroModel().aileron_limit
                            # limit = 1.0
                            )

        self.pid_pitch = PID(kp=0.0, ki=0.0, kd=0.0,
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


    def apply_action(self, action: np.ndarray) -> None:
        # doesn't have a direct effect in the simulation for roll and pitch
        # just sets the properties accordingly (useful for telemetry and logging)
        # and contrains the PI controller for throttle (maintain airspeed at 55 kph)
        super().apply_action(action)

        # apply the action (pitch and roll PID gains)
        self.pid_roll.set_gains(kp=self.sim[prp.kp_roll], ki=self.sim[prp.ki_roll], kd=self.sim[prp.kd_roll])

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