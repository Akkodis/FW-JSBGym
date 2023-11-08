import gymnasium as gym
import numpy as np
import csv
import yaml

from jsbgym.envs.jsbsim_env import JSBSimEnv
from typing import Type, NamedTuple, Tuple, Deque, Dict
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty
from collections import namedtuple, deque


class AttitudeControlTaskEnv(JSBSimEnv):
    """
        gym.Env wrapper task. Made for attitude control as described in Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy Optimization by Bohn et al.

        Attr:
            - `state_vars`: Tuple of BoundedProperty objects, defining the structure of an aircraft state (to be observed by the agent)
            - `action_vars`: Tuple of BoundedProperty objects, defining the structure of the action variables representing the control surface commands
            - `target_state_vars`: Tuple of BoundedProperty objects, defining the target state variables representing the reference setpoint for the controller to track
            - `telemetry_vars`: Tuple of BoundedProperty objects, defining the telemetry state variables representing the state of the aircraft to be logged
            - `State`: NamedTuple type, state of the aircraft
            - `TargetState`: NamedTuple type, target state of the aircraft
            - `steps_left`: BoundedProperty object, representing the number of steps left in the current episode
            - `aircraft_id`: Aircraft to simulate
            - `obs_history_size`: the size of the observation history, i.e. the number of previous states to be observed by the agent
            - `observation`: Deque of State NamedTuple objects, representing the observation of the agent
            - `flight_data_logfile`: the name of the file containing the flight data to be logged
            - `telemetry_fieldnames`: Tuple of strings, representing the fieldnames of the flight data to be logged for csv logging

    """
    state_vars: Tuple[BoundedProperty, ...] = (
        prp.airspeed_mps, # airspeed
        prp.roll_rad, prp.pitch_rad, # attitude
        prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
        prp.airspeed_err, prp.roll_err, prp.pitch_err, # errors
        prp.elevator_avg, prp.aileron_avg, prp.throttle_avg, # average of past 5 fcs commands
    )

    action_vars: Tuple[BoundedProperty, ...] = (
        prp.elevator_cmd, prp.aileron_cmd, # control surface commands normalized [-1, 1]
        prp.throttle_cmd # throttle command normalized [0, 1]
    )

    target_state_vars: Tuple[BoundedProperty, ...] = (
        prp.target_airspeed_mps, # target airspeed
        prp.target_roll_rad, prp.target_pitch_rad # target attitude
    )

    telemetry_vars: Tuple[BoundedProperty, ...] = (
        prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_m, # position
        prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
        prp.p_radps, prp.q_radps, prp.r_radps, prp.airspeed_mps, # angular rates and airspeed
        prp.throttle_cmd, prp.elevator_cmd, prp.aileron_cmd, # control surface commands
        prp.reward_total, prp.reward_airspeed, prp.reward_roll, prp.reward_pitch, prp.reward_actvar # rewards
    ) + target_state_vars # target state variables

    error_vars: Tuple[BoundedProperty, ...] = (
        prp.airspeed_err, prp.roll_err, prp.pitch_err # errors
    )

    State: Type[NamedTuple]
    TargetState: Type[NamedTuple]
    Errors: Type[NamedTuple]

    def __init__(self, config_file: str, render_mode: str=None) -> None:
        """
            Args:
                - `fdm_freq`: jsbsim FDM frequency
                - `obs_history_size`: the size of the observation history, i.e. the number of previous states to be observed by the agent
                - `flight_data_logfile`: the name of the file containing the flight data to be logged
        """
        # load config file
        with open(config_file, "r") as file:
            cfg_all: dict = yaml.safe_load(file)

        super().__init__(cfg_all["JSBSimEnv"], render_mode, 'x8')

        self.task_cfg: dict = cfg_all["AttitudeControlTaskEnv"]

        self.flight_data_logfile: str = self.task_cfg["flight_data_logfile"]

        self.obs_is_matrix = self.task_cfg["obs_is_matrix"]

        # observation history size
        self.obs_history_size: int = self.task_cfg["obs_history_size"]
        self.act_history_size: int = self.task_cfg["act_history_size"]

        # declaring state NamedTuple structure
        self.State: NamedTuple = namedtuple('State', [state_var.get_legal_name() for state_var in self.state_vars])
        self.state: self.State = None

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation_deque: Deque[np.ndarray] = deque(maxlen=self.obs_history_size) # deque of 1D nparrays containing self.State
        self.observation: np.ndarray = None # observation of the agent in a numpy array format

        # declaring action history. Deque with a maximum length of act_history_size
        self.action_hist: Deque[np.ndarray] = deque(maxlen=self.act_history_size) # action type: np.ndarray

        # declaring target state NamedTuple structure
        self.TargetState: NamedTuple = namedtuple('TargetState', [f"target_{t_state_var.get_legal_name()}" for t_state_var in self.target_state_vars])
        self.target: self.TargetState = None

        # declaring error NamedTuple structure
        self.Errors: NamedTuple = namedtuple('Errors', [f"{error_var.get_legal_name()}_err" for error_var in self.error_vars])
        self.errors: self.Errors = None

        self.ErrorSuccessTholds: NamedTuple = namedtuple('ErrorThresholds', [f"{error_var.get_legal_name()}_err_threshold" for error_var in self.error_vars])
        err_th_cfg: dict = self.task_cfg["error_success_tholds"]
        self.err_success_th = self.ErrorSuccessTholds(err_th_cfg["Va"], err_th_cfg["pitch"], err_th_cfg["roll"]) # error thresholds for airspeed, roll and pitch

        self.success_time_s: float = self.task_cfg["success_time_s"] # time in seconds the agent has to reach the target state to be considered successful
        self.reached_tsteps: int = 0 # number of timesteps the agent has reached the target state

        # create and set up csv logging file with flight telemetry
        self.telemetry_fieldnames: Tuple[str, ...] = tuple([prop.get_legal_name() for prop in self.telemetry_vars])
        with open(self.flight_data_logfile, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self.telemetry_fieldnames)
            csv_writer.writeheader()

        # set action and observation space from the task
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()


    def reset(self, seed: int=None, options: dict=None) -> np.ndarray:
        """
            Reset the task to its initial conditions.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        super().reset(seed=seed)

        self.reset_target_state() # reset task target state
        self.update_errors() # reset task errors
        self.update_action_history() # reset action history
        self.update_action_avg() # reset action avg
        self.sim[self.steps_left] = self.steps_left.max # reset the number of steps left in the episode to the max

        # reset observation and return the first observation of the episode
        self.observation_deque.clear()
        obs: np.ndarray = self.observe_state(first_obs=True)

        # compute 1st step reward (only to populate the properties for appropriate logging), not used in the optim
        # self.reward() 

        self.render() # render the simulation
        return obs, {}


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        # check if the action is valid
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")

        # apply the action to the simulation
        for prop, command in zip(self.action_vars, action):
            self.sim[prop] = command
        self.update_action_history(action) # update the action history

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()
            # convert the airspeed from kts to m/s at each sim step in the jsbsim properties
            self.convert_airspeed_kts2mps()
            # write the telemetry to a log csv file every fdm step (as opposed to every agent step -> to put out of this for loop)
            # self.flight_data_logging()
            # decrement the steps left
            self.sim[self.steps_left] -= 1

        # update the action_avg
        self.update_action_avg()

        # update the errors
        self.update_errors()

        # get the state
        obs: np.ndarray = self.observe_state()

        # get the reward
        self.reward: float = self.get_reward_bohn(action)

        # check if the episode is terminated modifies the reward with extra penalty if necessary
        terminated = self.is_terminated()
        truncated, episode_end, out_of_bounds = self.is_truncated()

        self.flight_data_logging()

        # info dict for debugging and misc infos
        info: Dict = {"steps_left": self.sim[self.steps_left],
                      "non_norm_obs": self.observation,
                      "non_norm_reward": self.reward,
                      "episode_end": episode_end,
                      "out_of_bounds": out_of_bounds
                    #   "crashed": crashed
                    }

        return obs, self.reward, terminated, truncated, info


    def is_terminated(self) -> Tuple[bool]:
        """
            Check if the episode is terminated, i.e. if the agent reaches the target state or crashes.
            Returns: False for now because no crash detection
        """
        # is_crashed: bool = self.sim[prp.altitude_sl_m] <= 0 # check collision with ground TODO: Remove
        # if is_crashed:
        #     reward -= 200 # penalize crash with -10 reward

        # is_target_reached: bool = False
        # np_err: np.ndarray = np.array(self.errors[:])
        # np_err_th: np.ndarray = np.array(self.err_success_th[:])
        # every timestep the agent has to reach the target state for 5 seconds, increment reached_tsteps
        # if np.all(np_err < np_err_th):
        #     self.reached_tsteps += 1
        # else: # else reset reached_tsteps
        #     self.reached_tsteps = 0

        # # if the agent has reached the target state for more than success_time seconds, return True
        # if self.reached_tsteps >= self.success_time_s * (1/self.sim.fdm_dt):
        #     is_target_reached = True

        return False


    def is_truncated(self) -> Tuple[bool, bool, bool]:
        """
            Check if the episode is truncated, i.e. if the episode reaches the maximum number of steps.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        episode_end: bool = self.sim[self.steps_left] <= 0 # if the episode is done, return True
        obs_out_of_bounds: bool = self.observation not in self.observation_space # if the observation contains out of bounds obs (due to JSBSim diverging), return True
        # if obs_out_of_bounds:
        #     reward -= 200

        # print((self.observation <= self.observation_space.high) & (self.observation >= self.observation_space.low))
        return episode_end or obs_out_of_bounds, episode_end, obs_out_of_bounds


    def update_action_history(self, action: np.ndarray=None) -> None:
        """
            Update the action history with the newest action and drop the oldest action.
            If it's the first action, the action history is initialized to `obs_history_size` * `action`.
        """
        # if it's the first action -> action is None: fill action history with [0, 0, 0]
        init_action: np.ndarray = np.array([0, 0, 0])
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
        state_lows: np.ndarray = np.array([state_var.min for state_var in self.state_vars], dtype=np.float32)
        state_highs: np.ndarray = np.array([state_var.max for state_var in self.state_vars], dtype=np.float32)

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


    def get_action_space(self) -> gym.spaces.Box:
        """
            Get the action space of the task.
        """
        # define action space
        action_lows: np.ndarray = np.array([action_var.min for action_var in self.action_vars], dtype=np.float32)
        action_highs: np.ndarray = np.array([action_var.max for action_var in self.action_vars], dtype=np.float32)
        action_space = gym.spaces.Box(low=action_lows, high=action_highs, dtype=np.float32)
        return action_space


    def observe_state(self, first_obs: bool = False) -> np.ndarray:
        """
            Observe the state of the aircraft, i.e. the state variables defined in the `state_vars` tuple, `obs_history_size` times.\\
            If it's the first observation, the observation is initialized to `obs_history_size` * `state`.\\
            Otherwise the observation is the newest `state` appended to the observation history and the oldest is dropped.
        """
        self.state = self.State(*[self.sim[prop] for prop in self.state_vars]) # create state named tuple with state variable values from the sim properties

        # if it's the first observation i.e. following a reset(): fill observation with obs_history_size * state
        if first_obs:
            for _ in range(self.obs_history_size):
                self.observation_deque.append(self.state)
        # else just append the newest state
        else:
            self.observation_deque.append(self.state)

        # return observation as a numpy array and add one channel dim for CNN policy
        if self.obs_is_matrix:
            self.observation: np.ndarray = np.expand_dims(np.array(self.observation_deque), axis=0).astype(np.float32)
        else:
            self.observation: np.ndarray = np.array(self.observation_deque).astype(np.float32)
        return self.observation


    def update_errors(self) -> None:
        """
            Update the error properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties
        self.sim[prp.airspeed_err] = self.sim[prp.target_airspeed_mps] - self.sim[prp.airspeed_mps]
        self.sim[prp.roll_err] = self.sim[prp.target_roll_rad] - self.sim[prp.roll_rad]
        self.sim[prp.pitch_err] = self.sim[prp.target_pitch_rad] - self.sim[prp.pitch_rad]
        
        # fill errors namedtuple with error variable values from the sim properties
        self.errors = self.Errors(*[self.sim[prop] for prop in self.error_vars])


    def update_action_avg(self) -> None:
        """
            Update the average of the past N commands (elevator, aileron, throttle)
        """
        self.sim[prp.elevator_avg] = np.mean(np.array(self.action_hist)[:, 0])
        self.sim[prp.aileron_avg] = np.mean(np.array(self.action_hist)[:, 1])
        self.sim[prp.throttle_avg] = np.mean(np.array(self.action_hist)[:, 2])


    def set_target_state(self, target_airspeed_mps: float, target_roll_rad: float, target_pitch_rad: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
        """
        # fill target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_airspeed_mps), str(target_roll_rad), str(target_pitch_rad))

        # set target state sim properties
        self.sim[prp.target_airspeed_mps] = target_airspeed_mps
        self.sim[prp.target_roll_rad] = target_roll_rad
        self.sim[prp.target_pitch_rad] = target_pitch_rad


    def reset_target_state(self) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset task class attributes with initial conditions
        self.set_target_state(target_airspeed_mps=self.sim[prp.initial_airspeed_kts] * 0.514444, # converting kts to mps 
                              target_roll_rad=self.sim[prp.initial_roll_rad], 
                              target_pitch_rad=self.sim[prp.initial_pitch_rad])


    def flight_data_logging(self) -> None:
        """
            Log flight data to csv.
        """
        # write flight data to csv
        with open(self.flight_data_logfile, 'a') as csv_file:
            csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=self.telemetry_fieldnames)
            info: dict[str, float] = {}
            for fieldname, prop in zip(self.telemetry_fieldnames, self.telemetry_vars):
                info[fieldname] = self.sim[prop]
            csv_writer.writerow(info)


    def get_reward(self) -> float:
        """
            Calculate the reward for the current step.\\
            Returns:
                - reward scalar: float
        """
        r_w: dict = self.task_cfg["reward_weights"] # reward weights for each reward component
        r_roll = np.clip(abs(self.sim[prp.roll_err]) / r_w["roll"]["scaling"], r_w["roll"]["clip_min"], r_w["roll"]["clip_max"]) # roll reward component
        r_pitch = np.clip(abs(self.sim[prp.pitch_err]) / r_w["pitch"]["scaling"], r_w["pitch"]["clip_min"], r_w["pitch"]["clip_max"]) # pitch reward component
        r_airspeed = np.clip(abs(self.sim[prp.airspeed_err]) / r_w["Va"]["scaling"], r_w["Va"]["clip_min"], r_w["Va"]["clip_max"]) # airspeed reward component

        # computing the cost attached to changing actuator setpoints to promote smooth non-oscillatory actuator behaviour
        diff: float = 0.0 # sum over all diffs between fcs commands of 2 consecutive timesteps
        r_actvar_raw: float = 0.0 # unclipped, unscaled fcs reward component
        # print(self.action_hist)
        # print(self.action_hist[2][1])
        # print(self.action_hist[-1][1])
        for fcs in range(len(self.action_vars)): # iterate through different fcs : [elevator: 0, aileron: 1, throttle: 2]
            for t in reversed(range(1, self.action_hist.maxlen)): # iterate through different timesteps backwards (avoid the t=0 case, for t-1=-1)
                diff += abs(self.action_hist[t][fcs] - self.action_hist[t-1][fcs]) # sum over all history of the command difference between t and t-1 for the same actuator
            r_actvar_raw += diff # sum those cmd diffs over all actuators
        r_actvar = np.clip(r_actvar_raw / r_w["act_var"]["scaling"], r_w["act_var"]["clip_min"], r_w["act_var"]["clip_max"]) # flight control surface reward component

        # return the negative sum of all reward components
        r_total = -(r_roll + r_pitch + r_airspeed + r_actvar) 

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_airspeed] = r_airspeed
        self.sim[prp.reward_actvar] = r_actvar
        self.sim[prp.reward_total] = r_total

        return r_total


    def get_reward_bohn(self, action: np.ndarray) -> float:
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
        # r_total: float = -(r_roll + r_pitch + r_airspeed + r_actvar + r_act_bounds) 
        r_total: float = -(r_roll + r_pitch + r_airspeed + r_act_bounds) 

        # populate properties
        self.sim[prp.reward_roll] = r_roll
        self.sim[prp.reward_pitch] = r_pitch
        self.sim[prp.reward_airspeed] = r_airspeed
        self.sim[prp.reward_actvar] = r_actvar
        self.sim[prp.reward_act_bounds] = r_act_bounds
        self.sim[prp.reward_total] = r_total

        return r_total