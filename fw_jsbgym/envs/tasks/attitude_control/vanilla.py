import numpy as np
from typing import Tuple
from omegaconf import DictConfig

from fw_jsbgym.envs.jsbsim_env import JSBSimEnv
from fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova import ACBohnNoVaTask, ACBohnNoVaIErrTask
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty

class ACVanillaTask(ACBohnNoVaTask):
    """
        Attitude control (without throttle control) task with minimal state and action space.
        No integral error, no wind states (alpha, beta), no past action nor action average.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err # errors
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err # errors
        )

        self.telemetry_prps += self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaYawTask(ACVanillaTask):
    """
        Attitude control (without throttle control) task with minimal state and action space + yaw state.
        No integral error, no wind states (alpha, beta), no past action nor action average.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.state_prps += (prp.heading_rad,) # add yaw to the state properties

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err # errors
        )

        self.telemetry_prps += self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)

class ACVanillaActTask(ACVanillaTask):
    """
        Same as the parent class, but addition of previous action in the current state.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # add previous action to the pre-existing state properties
        self.state_prps += (prp.aileron_cmd, prp.elevator_cmd)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaAlphaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of the angle of attack in the state space.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # add alpha to the pre-existing state properties
        self.state_prps += (prp.alpha_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaBetaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of the sideslip angle in the state space.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # add beta to the pre-existing state properties
        self.state_prps += (prp.beta_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaAlphaBetaTask(ACVanillaTask):
    """
        Same as the parent class, but with addition of both AoA and sideslip angles in the state space.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # add alpha and beta to the pre-existing state properties
        self.state_prps += (prp.alpha_rad, prp.beta_rad,)

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaThrTask(ACBohnNoVaTask):
    """
        Attitude control (roll and pitch) task with minimal state space.
        No integral error, no wind states (alpha, beta), no past action nor action average.
        No airspeed reference is fed in the state space, only roll and pitch. But contrary to the parent class,
        and to the ACBohnNoVaTask(s), the agent has to control on the throttle (instead of being a PI)
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # redefine the state properties entirely to a minimal state space
        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err # errors
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.aileron_cmd, prp.elevator_cmd, # control surface commands normalized [-1, 1]
            prp.throttle_cmd # throttle command normalized [0, 1]
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


    def apply_action(self, action: np.ndarray) -> None:
        # Use the apply action method from the JSBSimEnv class 
        # (not the direct parent class which adds up the output of a PI controller for the airspeed)
        return JSBSimEnv.apply_action(self, action)


class ACVanillaIErrTask(ACBohnNoVaIErrTask):
    """
        Attitude control task with vanilla state and action space.
        But with integral error states.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # rewriting ACVanilla state space + integral errors
        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, # errors
            prp.roll_integ_err, prp.pitch_integ_err, # integral errors
        )

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaActIErrTask(ACVanillaIErrTask):
    """
        Attitude control task with vanilla state + last action
        + integral error states.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # rewriting ACVanilla state space + integral errors
        self.state_prps += (prp.aileron_cmd, prp.elevator_cmd,)

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaActIErrAlphaTask(ACVanillaActIErrTask):
    """
        Attitude control task with vanilla state + last action
        + integral error states and alpha in the state space.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # rewriting ACVanilla state space + integral errors + alpha
        self.state_prps += (prp.alpha_rad,)

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)


class ACVanillaActIErrBetaTask(ACVanillaActIErrTask):
    """
        Attitude control task with vanilla state + last action
        + integral error states and beta in the state space.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        super().__init__(cfg_env, telemetry_file, render_mode)

        # rewriting ACVanilla state space + integral errors + beta
        self.state_prps += (prp.beta_rad,)

        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps # target state variables

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        
        self.telemetry_setup(self.telemetry_file)

