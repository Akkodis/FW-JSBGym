import gymnasium as gym
import numpy as np
import yaml
from typing import Tuple, Deque, Dict
from collections import deque

from jsbgym.envs.jsbsim_env import JSBSimEnv
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty


class AttitudeControlTask(JSBSimEnv):
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
    def __init__(self, config_file: str, telemetry_file: str='', render_mode: str='none') -> None:
        """
            Args:
                - `config_file`: the name of the config file containing the task parameters
                - `telemetry_file`: the name of the file containing the flight data to be logged
                - `render_mode`: the render mode for the task
        """
        # load config file
        with open(config_file, "r") as file:
            cfg_all: dict = yaml.safe_load(file)

        super().__init__(cfg_all["JSBSimEnv"], telemetry_file, render_mode)

        self.task_cfg: dict = cfg_all["AttitudeControlTask"]

        self.obs_is_matrix = self.task_cfg["obs_is_matrix"]

        # observation history size
        self.obs_history_size: int = self.task_cfg["obs_history_size"]
        self.act_history_size: int = self.task_cfg["act_history_size"]

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

        self.telemetry_prps: Tuple[BoundedProperty, ...] = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
            prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
            prp.p_radps, prp.q_radps, prp.r_radps, # angular rates and airspeed
            prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, # control surface commands
            prp.reward_total, prp.reward_roll, prp.reward_pitch, prp.reward_airspeed, # rewards
            prp.airspeed_mps, prp.airspeed_kph, # airspeed
            prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps, # wind speed mps
            prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph, # wind speed kph
            prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps, # turbulence mps
            prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph, # turbulence kph
        ) + self.target_prps # target state variables

        self.error_prps: Tuple[BoundedProperty, ...] = (
            prp.roll_err, prp.pitch_err, prp.airspeed_err # errors
        )

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation_deque: Deque[np.ndarray] = deque(maxlen=self.obs_history_size) # deque of 1D nparrays containing self.State

        # declaring action history. Deque with a maximum length of act_history_size
        self.action_hist: Deque[np.ndarray] = deque(maxlen=self.act_history_size) # action type: np.ndarray

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.initialize()
        self.telemetry_setup(self.telemetry_file)


    def reset(self, seed: int=None, options: dict=None) -> np.ndarray:
        """
            Reset the task to its initial conditions.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        super().reset(seed=seed, options=options)

        self.reset_target_state() # reset task target state
        self.update_errors() # reset task errors
        self.update_action_history() # reset action history
        self.update_action_avg() # reset action avg
        self.sim[self.steps_left] = self.steps_left.max # reset the number of steps left in the episode to the max
        self.sim[self.current_step] = self.current_step.min # reset the number of steps left in the episode to 

        # reset observation and return the first observation of the episode
        self.observation_deque.clear()
        obs: np.ndarray = self.observe_state(first_obs=True)

        self.render() # render the simulation
        return obs, {}


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
            Steps the task forward.
        """
        # check if the action is valid
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")

        # apply the action to the simulation
        self.apply_action(action)
        self.update_action_history(action) # update the action history

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()
            # write the telemetry to a log csv file every fdm step (as opposed to every agent step -> to put out of this for loop)
            # self.telemetry_logging()
            # decrement the steps left
            self.sim[self.steps_left] -= 1
            self.sim[self.current_step] += 1

        # update the action_avg
        self.update_action_avg()

        # update the errors
        self.update_errors()

        # get the state
        self.observation = self.observe_state()

        # get the reward
        self.reward: float = self.get_reward(action)

        # check if the episode is terminated modifies the reward with extra penalty if necessary
        terminated = self.is_terminated()
        truncated, episode_end, out_of_bounds = self.is_truncated()

        # write telemetry to a csv file every agent step
        if self.render_mode in self.metadata["render_modes"][1:]:
            self.telemetry_logging()

        # info dict for debugging and misc infos
        info: Dict = {"steps_left": self.sim[self.steps_left],
                      "non_norm_obs": self.observation,
                      "non_norm_reward": self.reward,
                      "episode_end": episode_end,
                      "out_of_bounds": out_of_bounds
                    }

        return self.observation, self.reward, terminated, truncated, info

    def apply_action(self, action: np.ndarray) -> None:
        # apply the action to the simulation
        for prop, command in zip(self.action_prps, action):
            self.sim[prop] = command

    def is_terminated(self) -> Tuple[bool]:
        """
            Check if the episode is terminated. In the current MDP formulation, there's no terminal state.
        """
        return False


    def is_truncated(self) -> Tuple[bool, bool, bool]:
        """
            Check if the episode is truncated, i.e. if the episode reaches the maximum number of steps or
            if the observation contains out of bounds obs (due to JSBSim diverging).
            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        episode_end: bool = self.sim[self.steps_left] <= 0 # if the episode is done, return True
        obs_out_of_bounds: bool = self.observation not in self.observation_space # if the observation contains out of bounds obs (due to JSBSim diverging), return True

        if obs_out_of_bounds:
            print(f"Out of bounds observation: {self.observation}")
            print(f"Turbulence: {self.sim[prp.turb_type]}")
            print(f"Turbulence: {self.sim[prp.turb_w20_fps]}")
            print(f"Turbulence: {self.sim[prp.turb_severity]}")

        return episode_end or obs_out_of_bounds, episode_end, obs_out_of_bounds


    def update_action_history(self, action: np.ndarray=None) -> None:
        """
            Update the action history with the newest action and drop the oldest action.
            If it's the first action, the action history is initialized to `obs_history_size` * `action`.
        """
        # if it's the first action -> action is None: fill action history with zeros
        init_action: np.ndarray = np.zeros(self.action_space.shape, dtype=np.float32)
        if action is None:
            for _ in range(self.obs_history_size):
                self.action_hist.append(init_action)
        # else just append the newest action
        else:
            self.action_hist.append(action)


    def get_observation_space(self) -> gym.spaces.Box:
        """
            Get the observation space of the task.
        """
        # defining observation space based on pre-chosen state variables
        state_lows: np.ndarray = np.array([state_var.min for state_var in self.state_prps], dtype=np.float32)
        state_highs: np.ndarray = np.array([state_var.max for state_var in self.state_prps], dtype=np.float32)

        # check if we want a matrix formatted observation space shape=(obs_history_size, state_vars) for CNN policy
        if self.obs_is_matrix:
            state_lows: np.ndarray = np.expand_dims(np.array([state_lows for _ in range(self.obs_history_size)]), axis=0)
            state_highs: np.ndarray = np.expand_dims(np.array([state_highs for _ in range(self.obs_history_size)]), axis=0)
            observation_space = gym.spaces.Box(low=np.array(state_lows), high=np.array(state_highs), dtype=np.float32)
        else: # else we want a vector formatted observation space len=(obs_history_size * state_vars) for MLP policy
            # multiply state_lows and state_highs by obs_history_size to get the observation space
            observation_space = gym.spaces.Box(low=np.tile(state_lows, self.obs_history_size),
                                            high=np.tile(state_highs, self.obs_history_size), 
                                            dtype=np.float32)
        return observation_space


    def observe_state(self, first_obs: bool = False) -> np.ndarray:
        """
            Observe the state of the aircraft, i.e. the state variables defined in the `state_vars` tuple, `obs_history_size` times.\\
            If it's the first observation, the observation is initialized to `obs_history_size` * `state`.\\
            Otherwise the observation is the newest `state` appended to the observation history and the oldest is dropped.
        """
        self.state = self.State(*[self.sim[prop] for prop in self.state_prps]) # create state named tuple with state variable values from the sim properties

        # if it's the first observation i.e. following a reset(): fill observation with obs_history_size * state
        if first_obs:
            for _ in range(self.obs_history_size):
                self.observation_deque.append(self.state)
        # else just append the newest state
        else:
            self.observation_deque.append(self.state)

        # return observation as a numpy array and add one channel dim for CNN policy
        if self.obs_is_matrix:
            obs: np.ndarray = np.expand_dims(np.array(self.observation_deque), axis=0).astype(np.float32)
        else:
            obs: np.ndarray = np.array(self.observation_deque).astype(np.float32)
        return obs


    def update_errors(self) -> None:
        """
            Update the error properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties
        self.sim[prp.roll_err] = self.sim[prp.target_roll_rad] - self.sim[prp.roll_rad]
        self.sim[prp.pitch_err] = self.sim[prp.target_pitch_rad] - self.sim[prp.pitch_rad]
        self.sim[prp.airspeed_err] = self.sim[prp.target_airspeed_kph] - self.sim[prp.airspeed_kph]

        # fill errors namedtuple with error variable values from the sim properties
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_prps])


    def update_action_avg(self) -> None:
        """
            Update the average of the past N commands (elevator, aileron, throttle)
        """
        self.sim[prp.aileron_avg] = np.mean(np.array(self.action_hist)[:, 0])
        self.sim[prp.elevator_avg] = np.mean(np.array(self.action_hist)[:, 1])
        self.sim[prp.throttle_avg] = np.mean(np.array(self.action_hist)[:, 2])


    def set_target_state(self, target_roll_rad: float, target_pitch_rad: float, target_airspeed_kph: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
        """
        # set target state sim properties
        self.sim[prp.target_roll_rad] = target_roll_rad
        self.sim[prp.target_pitch_rad] = target_pitch_rad
        self.sim[prp.target_airspeed_kph] = target_airspeed_kph

        # fill target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_roll_rad), str(target_pitch_rad), str(target_airspeed_kph))


    def reset_target_state(self) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset task class attributes with initial conditions
        self.set_target_state(target_roll_rad=self.sim[prp.initial_roll_rad], 
                              target_pitch_rad=self.sim[prp.initial_pitch_rad],
                              target_airspeed_kph=self.sim[prp.initial_airspeed_kts] * 1.852) # converting kts to kph


    def get_reward(self, action: np.ndarray) -> float:
        """
            Reward function
            Based on the bohn PPO paper reward func, but without the actvar component (taken care by CAPS loss)
        """
        r_w: dict = self.task_cfg["reward_weights"] # reward weights for each reward component
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
        deltas: np.ndarray = np.diff(np_action_hist[-self.obs_history_size:], axis=0)
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
