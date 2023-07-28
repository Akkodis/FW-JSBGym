import gymnasium as gym
import numpy as np
import math
import csv


from abc import ABC
from simulation.jsb_simulation import Simulation
from typing import Type, NamedTuple, Tuple, Deque, Dict
from utils import jsbsim_properties as prp
from utils.jsbsim_properties import BoundedProperty
from collections import namedtuple, deque


class Task(ABC):
    """
        Interface class, for being the base class of future multiple task classes.
    """
    ...

class AttitudeControlTask(Task, ABC):
    """
        gym.Env wrapper task. Made for attitude control as described in Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy Optimization by Bohn et al.

        Attr:
            - `state_vars`: Tuple of BoundedProperty objects, defining the structure of an aircraft state (to be observed by the agent)
            - `action_vars`: Tuple of BoundedProperty objects, defining the structure of the action variables representing the control surface commands
            - `target_state_vars`: Tuple of BoundedProperty objects, defining the target state variables representing the reference setpoint for the controller to track
            - `telemetry_vars`: Tuple of BoundedProperty objects, defining the telemetry state variables representing the state of the aircraft to be logged
            - `State`: NamedTuple type, state of the aircraft
            - `TargetState`: NamedTuple type, target state of the aircraft
            - `episode_time_s`: the duration of the episode in seconds
            - `steps_left`: BoundedProperty object, representing the number of steps left in the current episode
            - `aircraft_id`: Aircraft to simulate
            - `obs_history_size`: the size of the observation history, i.e. the number of previous states to be observed by the agent
            - `observation`: Deque of State NamedTuple objects, representing the observation of the agent
            - `flight_data_logfile`: the name of the file containing the flight data to be logged
            - `telemetry_fieldnames`: Tuple of strings, representing the fieldnames of the flight data to be logged for csv logging

    """
    DEFAULT_EPISODE_TIME_S = 60.0

    state_vars: Tuple[BoundedProperty, ...] = (
        prp.airspeed_kts, # airspeed
        prp.roll_rad, prp.pitch_rad, # attitude
        prp.p_radps, prp.q_radps, prp.r_radps, # angular rates
        prp.airspeed_err, prp.roll_err, prp.pitch_err # errors
    )

    action_vars: Tuple[BoundedProperty, ...] = (
        prp.elevator_cmd, prp.aileron_cmd, # control surface commands normalized [-1, 1]
        prp.throttle_cmd # throttle command normalized [0, 1]
    )

    target_state_vars: Tuple[BoundedProperty, ...] = (
        prp.target_airspeed_kts, # target airspeed
        prp.target_roll_rad, prp.target_pitch_rad # target attitude
    )

    telemetry_vars: Tuple[BoundedProperty, ...] = (
        prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_ft, # position
        prp.roll_rad, prp.pitch_rad, prp.heading_rad, # attitude
        prp.p_radps, prp.q_radps, prp.r_radps, prp.airspeed_kts, # angular rates and airspeed
        prp.throttle_cmd, prp.elevator_cmd, prp.aileron_cmd, # control surface commands
    ) + target_state_vars # target state variables

    error_vars: Tuple[BoundedProperty, ...] = (
        prp.airspeed_err, prp.roll_err, prp.pitch_err # errors
    )

    State: Type[NamedTuple]
    TargetState: Type[NamedTuple]
    Errors: Type[NamedTuple]

    def __init__(self, fdm_freq: float, obs_history_size: int, flight_data_logfile: str = "data/flight_data.csv", episode_time_s: float = DEFAULT_EPISODE_TIME_S):
        """
            Args:
                - `fdm_freq`: jsbsim FDM frequency
                - `obs_history_size`: the size of the observation history, i.e. the number of previous states to be observed by the agent
                - `flight_data_logfile`: the name of the file containing the flight data to be logged
                - `episode_time_s`: the duration of an episode in seconds
        """
        # set episode time
        self.episode_time_s: float = episode_time_s
        max_episode_steps: int = math.ceil(episode_time_s * fdm_freq)
        self.steps_left: BoundedProperty = BoundedProperty("info/steps_left", "steps remaining in the current episode", 0, max_episode_steps)

        # observation history size
        self.obs_history_size: int = obs_history_size

        # declaring state NamedTuple structure
        self.State: NamedTuple = namedtuple('State', [state_var.get_legal_name() for state_var in self.state_vars])
        self.state: self.State = None

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation: Deque[self.State] = deque(maxlen=self.obs_history_size) # self.State type: NamedTuple

        # declaring action history. Deque with a maximum length of obs_history_size (action history size = observation history size)
        self.action_hist: Deque[np.ndarray] = deque(maxlen=self.obs_history_size) # self.State type: NamedTuple

        # declaring target state NamedTuple structure
        self.TargetState: NamedTuple = namedtuple('TargetState', [f"target_{t_state_var.get_legal_name()}" for t_state_var in self.target_state_vars])
        self.target: self.TargetState = None

        # declaring error NamedTuple structure
        self.Errors: NamedTuple = namedtuple('Errors', [f"{error_var.get_legal_name()}_err" for error_var in self.error_vars])
        self.errors: self.Errors = None

        # create and set up csv logging file with flight telemetry
        self.flight_data_logfile: str = flight_data_logfile
        self.telemetry_fieldnames: Tuple[str, ...] = tuple([prop.get_legal_name() for prop in self.telemetry_vars])
        with open(self.flight_data_logfile, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self.telemetry_fieldnames)
            csv_writer.writeheader()


    def reset_task(self, sim: Simulation) -> np.ndarray:
        """
            Reset the task to its initial conditions.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        self.reset_target_state(sim) # reset task target state
        self.update_errors(sim) # reset task errors
        self.update_action_history() # reset action history
        sim[self.steps_left] = self.steps_left.max # reset the number of steps left in the episode to the max

        # reset observation and return the first observation of the episode
        self.observation.clear()
        obs: np.ndarray = self.observe_state(sim, first_obs=True)
        return obs


    def step_task(self, sim: Simulation, action: np.ndarray, sim_steps_after_agent_action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # apply the action to the simulation
        for prop, command in zip(self.action_vars, action):
            sim[prop] = command
        self.update_action_history(action) # update the action history

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(sim_steps_after_agent_action):
            sim.run_step()
            # write the telemetry to a log csv file every fdm step (as opposed to every agent step -> to put out of this for loop)
            self.flight_data_logging(sim)
            # decrement the steps left
            sim[self.steps_left] -= 1

        # get the state
        obs: np.ndarray = self.observe_state(sim)

        # update the errors
        self.update_errors(sim)

        # get the reward
        reward: float = self.reward(sim)

        # check if the episode is done
        done: bool = self.is_terminal(sim)

        # info dict for debugging and misc infos
        info: Dict = {"steps_left": sim[self.steps_left],
                      "reward": reward}

        return obs, reward, done, info
    

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


    def is_terminal(self, sim: Simulation) -> bool:
        """
            Check if the current step is terminal, i.e. if the episode is done.

            Args:
                - `sim`: the simulation object containing the JSBSim FDM
        """
        is_terminal_step: bool = sim[self.steps_left] <= 0 # if the episode is done, return True
        is_crashed: bool = sim[prp.altitude_sl_ft] <= 0 # check collision with ground
        return is_terminal_step or is_crashed


    def get_observation_space(self) -> gym.spaces.Box:
        """
            Get the observation space of the task.
        """
        # defining observation space based on pre-chosen state variables
        state_lows: np.ndarray = np.array([state_var.min for state_var in self.state_vars])
        state_highs: np.ndarray = np.array([state_var.max for state_var in self.state_vars])

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
        action_lows: np.ndarray = np.array([action_var.min for action_var in self.action_vars])
        action_highs: np.ndarray = np.array([action_var.max for action_var in self.action_vars])
        action_space = gym.spaces.Box(low=action_lows, high=action_highs, dtype=np.float32)
        return action_space


    def observe_state(self, sim: Simulation, first_obs: bool = False) -> np.ndarray:
        """
            Observe the state of the aircraft, i.e. the state variables defined in the `state_vars` tuple, `obs_history_size` times.\\
            If it's the first observation, the observation is initialized to `obs_history_size` * `state`.\\
            Otherwise the observation is the newest `state` appended to the observation history and the oldest is dropped.
        """
        self.state = self.State(*[sim[prop] for prop in self.state_vars]) # create state named tuple with state variable values from the sim properties

        # if it's the first observation i.e. following a reset(): fill observation with obs_history_size * state
        if first_obs:
            for _ in range(self.obs_history_size):
                self.observation.append(self.state)
        # else just append the newest state
        else:
            self.observation.append(self.state)

        # return observation as a numpy array
        obs_nparray: np.ndarray = np.array(self.observation).flatten()
        return obs_nparray


    def update_errors(self, sim: Simulation) -> None:
        """
            Update the error properties of the aircraft, i.e. the difference between the target state and the current state.
        """
        # update error sim properties
        sim[prp.airspeed_err] = sim[prp.target_airspeed_kts] - sim[prp.airspeed_kts]
        sim[prp.roll_err] = sim[prp.target_roll_rad] - sim[prp.roll_rad]
        sim[prp.pitch_err] = sim[prp.target_pitch_rad] - sim[prp.pitch_rad]
        
        # fill errors namedtuple with error variable values from the sim properties
        self.errors = self.Errors(*[sim[prop] for prop in self.error_vars])


    def set_target_state(self, sim: Simulation, target_airspeed_kts: float, target_roll_rad: float, target_pitch_rad: float) -> None:
        """
            Set the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple.
        """
        # fill target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_airspeed_kts), str(target_roll_rad), str(target_pitch_rad))

        # set target state sim properties
        sim[prp.target_airspeed_kts] = target_airspeed_kts
        sim[prp.target_roll_rad] = target_roll_rad
        sim[prp.target_pitch_rad] = target_pitch_rad


    def reset_target_state(self, sim: Simulation) -> None:
        """
            Reset the target state of the aircraft, i.e. the target state variables defined in the `target_state_vars` tuple, with initial conditions.
        """
        # reset task class attributes with initial conditions
        self.set_target_state(sim, target_airspeed_kts=sim[prp.initial_airspeed_kts], 
                              target_roll_rad=sim[prp.initial_roll_rad], 
                              target_pitch_rad=sim[prp.initial_pitch_rad])


    def flight_data_logging(self, sim: Simulation) -> None:
        """
            Log flight data to csv.
        """
        # write flight data to csv
        with open(self.flight_data_logfile, 'a') as csv_file:
            csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=self.telemetry_fieldnames)
            info: dict[str, float] = {}
            for fieldname, prop in zip(self.telemetry_fieldnames, self.telemetry_vars):
                info[fieldname] = sim[prop]
            csv_writer.writerow(info)


    def reward(self, sim:Simulation) -> float:
        """
            Calculate the reward for the current step.\\
            Returns:
                - reward scalar: float
        """
        r_roll = np.clip(abs(sim[prp.roll_err]) / 3.3, 0, 0.3) # roll reward component
        r_pitch = np.clip(abs(sim[prp.roll_err]) / 2.25, 0, 0.3) # pitch reward component
        r_airspeed = np.clip(abs(sim[prp.roll_err]) / 25, 0, 0.3) # airspeed reward component

        # computing the cost attached to changing actuator setpoints to promote smooth non-oscillatory actuator behaviour
        diff: float = 0.0 # sum over all diffs between fcs commands of 2 consecutive timesteps
        r_fcs_raw: float = 0.0 # unclipped, unscaled fcs reward component
        for fcs in range(self.action_hist[1].shape[0]): # iterate through different fcs : [elevator: 0, aileron: 1, throttle: 2]
            for t in reversed(range(self.action_hist[0].shape[0])): # iterate through different timesteps backwards
                diff += abs(self.action_hist[t][fcs] - self.action_hist[t-1][fcs]) # sum over all history of the command difference between t and t-1 for the same actuator
            r_fcs_raw += diff # sum those cmd diffs over all actuators
        r_fcs = np.clip(r_fcs_raw / 60, 0, 0.1) # flight control surface reward component
        return -(r_roll + r_pitch + r_airspeed + r_fcs)
