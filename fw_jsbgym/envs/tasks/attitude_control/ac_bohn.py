import gymnasium as gym
import numpy as np
from typing import Tuple
from omegaconf import DictConfig

from fw_jsbgym.envs.tasks.jsbsim_task import JSBSimTask
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty


class ACBohnTask(JSBSimTask):
    """
        gym.Env wrapper task. Made for attitude control as described in Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy Optimization by Bohn et al.

        Attr:
            Task specific attributes:
            - `state_prps`: Tuple of BoundedProperty objects, defining the structure of an aircraft state (to be observed by the agent)
            - `action_prps`: Tuple of BoundedProperty objects, defining the structure of the action variables representing the control surface commands
            - `target_prps`: Tuple of BoundedProperty objects, defining the target state variables representing the reference setpoint for the controller to track
            - `telemetry_prps`: Tuple of BoundedProperty objects, defining the telemetry state variables representing the state of the aircraft to be logged
            - `error_prps`: Tuple of BoundedProperty objects, defining the structure of the error variables representing the difference between the target state and the current state
            - `obs_history_size`: the size of the observation history, i.e. the number of previous states to be observed by the agent
            - `act_history_size`: the size of the action history, i.e. the number of previous actions to be observed by the agent
            - `observation_deque`: Deque of State NamedTuple objects, representing the past observations of the agent
            - `action_hist`: Deque of np.ndarray objects, representing the past actions of the agent
            - `observation_space`: the observation space of the task
            - `action_space`: the action space of the task

    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode: str='none') -> None:
        """
            Args:
                - `cfg_env`: DictConfig object containing the environment configuration
                - `telemetry_file`: the name of the file containing the flight data to be logged
                - `render_mode`: the render mode for the task
        """
        super().__init__(cfg_env, telemetry_file, render_mode)

        self.task_cfg: DictConfig = cfg_env.task

        self.state_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_rad, prp.pitch_rad, # attitude
            prp.airspeed_kph, # airspeed
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
            prp.roll_err, prp.pitch_err, prp.airspeed_err, # errors
            prp.elevator_avg, prp.aileron_avg, prp.throttle_avg # average of past 5 fcs commands
        )

        self.action_prps: Tuple[BoundedProperty, ...] = (
            prp.aileron_cmd, prp.elevator_cmd, # control surface commands normalized [-1, 1]
            prp.throttle_cmd # throttle command normalized [0, 1]
        )

        self.target_prps: Tuple[BoundedProperty, ...] = (
            prp.target_roll_rad, prp.target_pitch_rad, # target attitude
            prp.target_airspeed_kph # target airspeed
        )

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, prp.airspeed_err # errors
        )

        self.reward_prps: Tuple[BoundedProperty, ...] = (
            prp.reward_roll, prp.reward_pitch, prp.reward_airspeed, # reward components
        )

        # telemetry properties are an addition of the common telemetry properties, target properties and error properties
        self.telemetry_prps = self.common_telemetry_prps + self.target_prps + self.error_prps + self.reward_prps

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        if self.jsbsim_cfg.debug and ACBohnTask.__name__ == self.__class__.__name__:
            self.print_MDP_info()

        self.telemetry_setup(self.telemetry_file)


    def reset(self, seed: int=None, options: dict=None) -> Tuple[np.ndarray, dict]:
        """
            Reset the task to its initial conditions.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        self.observation, info = super().reset(seed=seed, options=options)

        # clear the observation deque (history of past observations)
        self.observation_deque.clear()

        return self.observation, info


    def reset_props(self) -> None:
        """
            Reset some of the properties (target, errors, action history, action averages, fcs position history etc...)
        """
        super().reset_props() # reset the parent class JSBSimEnv properties

        self.update_action_avg() # reset action avg


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
            Steps the task forward.
        """
        # update the action_avg
        self.update_action_avg()

        # update the action history
        self.update_action_history(action)

        # step the parent class JSBSimEnv where gusts are generated
        self.observation, self.reward, terminated, truncated, info = super().step(action)

        return self.observation, self.reward, terminated, truncated, info


    def update_errors(self, first_err=False) -> None:
        """
            Update the error properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties
        self.sim[prp.roll_err] = self.sim[prp.target_roll_rad] - self.sim[prp.roll_rad]
        self.sim[prp.pitch_err] = self.sim[prp.target_pitch_rad] - self.sim[prp.pitch_rad]
        self.sim[prp.airspeed_err] = self.sim[prp.target_airspeed_kph] - self.sim[prp.airspeed_kph]


    def update_action_avg(self) -> None:
        """
            Update the average of the past N commands (elevator, aileron, throttle)
        """
        self.sim[prp.aileron_avg] = np.mean(np.array(self.action_hist)[:, 0])
        self.sim[prp.elevator_avg] = np.mean(np.array(self.action_hist)[:, 1])
        self.sim[prp.throttle_avg] = np.mean(np.array(self.action_hist)[:, 2])


    def set_target_state(self, target_state: np.ndarray) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
            Arg: np.ndarray[target roll, target pitch and target airspeed]. Units: [rad, rad, kph].
        """
        # check that the target state has the correct shape
        if target_state.shape[0] != len(self.target_prps):
            raise ValueError(f"Target state should be a 1D ndarray of length {len(self.target_prps)} but got shape {target_state.shape}")

        # set target state sim properties
        self.sim[prp.target_roll_rad] = target_state[0]
        self.sim[prp.target_pitch_rad] = target_state[1]
        self.sim[prp.target_airspeed_kph] = target_state[2]


    def reset_target_state(self) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset task class attributes with initial conditions
        self.set_target_state(np.array([self.sim[prp.ic_roll_rad],
                                          self.sim[prp.ic_pitch_rad],
                                          self.sim[prp.ic_airspeed_kts] * 1.852])) # converting kts to kph


    def reset_ext_state_props(self):
        pass


    def get_reward(self, action: np.ndarray) -> float:
        """
            Reward function
            Based on the bohn PPO paper reward func, but without the actvar component (taken care by CAPS loss)
        """
        r_w: dict = self.task_cfg.reward.weights # reward weights for each reward component
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], r_w["roll"]["clip_min"], r_w["roll"].get("clip_max", None)) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], r_w["pitch"]["clip_min"], r_w["pitch"].get("clip_max", None)) # pitch reward component
        r_airspeed = np.clip(abs(self.sim[prp.airspeed_err]) / r_w["Va"]["scaling"], r_w["Va"]["clip_min"], r_w["Va"].get("clip_max", None)) # airspeed reward component

        # return the negative sum of all reward components
        r_total: float = -(r_roll + r_pitch + r_airspeed)

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_airspeed] = r_airspeed
        self.sim[prp.reward_total] = r_total

        return r_total


    def get_reward_bohn(self, action: np.ndarray) -> float:
        """
            Original reward function from the bohn PPO paper (results in lots of action oscillations / variation)
        """
        r_w: dict = self.task_cfg["reward_weights"] # reward weights for each reward component
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], r_w["roll"]["clip_min"], r_w["roll"].get("clip_max", None)) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], r_w["pitch"]["clip_min"], r_w["pitch"].get("clip_max", None)) # pitch reward component
        r_airspeed = np.clip(abs(self.sim[prp.airspeed_err]) / r_w["Va"]["scaling"], r_w["Va"]["clip_min"], r_w["Va"].get("clip_max", None)) # airspeed reward component

        r_act_low: np.ndarray = np.where(action < self.action_space.low, self.action_space.low - action, 0)
        r_act_high: np.ndarray = np.where(action > self.action_space.high, action - self.action_space.high, 0)
        r_act_bounds_raw: float = np.sum(np.abs(r_act_low) + np.sum(np.abs(r_act_high))) # doute sur le np.sum
        r_act_bounds: float = np.clip(r_act_bounds_raw / r_w["act_bounds"]["scaling"], 0, r_w["act_bounds"].get("clip_max", None))

        np_action_hist: np.ndarray = np.array(self.action_hist)
        deltas: np.ndarray = np.diff(np_action_hist[-self.task_cfg.mdp.obs_hist_size:], axis=0)
        r_actvar_raw = np.sum(np.abs(deltas))
        r_actvar = np.clip(r_actvar_raw / r_w["act_var"]["scaling"], r_w["act_var"]["clip_min"], r_w["act_var"].get("clip_max", None)) # flight control surface reward component

        # return the negative sum of all reward components
        r_total: float = -(r_roll + r_pitch + r_airspeed + r_actvar + r_act_bounds) 

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_airspeed] = r_airspeed
        self.sim[prp.reward_actvar] = r_actvar
        self.sim[prp.reward_act_bounds] = r_act_bounds
        self.sim[prp.reward_total] = r_total

        return r_total
