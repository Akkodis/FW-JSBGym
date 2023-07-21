import gymnasium as gym
import numpy as np
import math
import csv


from abc import ABC, abstractmethod
from simulation.jsb_simulation import Simulation
from typing import Type, NamedTuple, Tuple, Dict
from utils import jsbsim_properties as prp
from utils.jsbsim_properties import BoundedProperty
from collections import namedtuple


class Task(ABC):
    ...

class AttitudeControlTask(Task, ABC):
    DEFAULT_EPISODE_TIME_S = 60.0
    prp_airspeed_err = BoundedProperty("error/airspeed-err", "airspeed error", 0, np.inf)
    prp_roll_err = BoundedProperty("error/roll-err", "roll error", -np.pi, np.pi)
    prp_pitch_err = BoundedProperty("error/pitch-err", "pitch error", -np.pi, np.pi)
    prp_target_airspeed_kts = BoundedProperty("target/airspeed", "desired airspeed", 0, np.inf)
    prp_target_roll_rad = BoundedProperty("target/roll-rad", "desired roll angle [rad]", -np.pi, np.pi)
    prp_target_pitch_rad = BoundedProperty("target/pitch-rad", "desired pitch angle [rad]", -np.pi, np.pi)

    state_vars: Tuple[BoundedProperty, ...] = (
        prp.airspeed_kts, # airspeed
        prp.roll_rad, prp.pitch_rad,
        prp.p_radps, prp.q_radps, prp.r_radps,
        prp_airspeed_err, prp_roll_err, prp_pitch_err
    )
    action_vars: Tuple[BoundedProperty, ...] = (
        prp.elevator_cmd, prp.aileron_cmd, # control surface commands normalized [-1, 1]
        prp.throttle_cmd # throttle command normalized [0, 1]
    )
    target_state_vars: Tuple[BoundedProperty, ...] = (
        prp_target_airspeed_kts,
        prp_target_roll_rad, prp_target_pitch_rad
    )
    State: Type[NamedTuple]
    TargetState: Type[NamedTuple]

    def __init__(self, aircraft_id: str, fdm_freq: float, flight_data_logfile: str = "data/flight_data.csv", episode_time_s: float = DEFAULT_EPISODE_TIME_S):
        self.episode_time_s: float = episode_time_s
        max_episode_steps: int = math.ceil(episode_time_s * fdm_freq)
        self.steps_left: BoundedProperty = BoundedProperty("info/steps_left", "steps remaining in the current episode", 0, max_episode_steps)
        self.aircraft_id: str = aircraft_id

        # create state NamedTuple structure
        self.State: NamedTuple = namedtuple('State', [state_var.get_legal_name() for state_var in self.state_vars])

        # create target state NamedTuple structure
        self.TargetState = namedtuple('TargetState', [f"target_{t_state_var.get_legal_name()}" for t_state_var in self.target_state_vars])

        # create and set up csv logging file with flight telemetry
        self.fieldnames: Tuple[str] = (
                            'latitude', 'longitude', 'altitude', 
                            'roll', 'pitch', 'course', 
                            'roll_rate', 'pitch_rate', 'yaw_rate', 'airspeed',
                            'throttle_cmd', 'elevator_cmd', 'aileron_cmd',
                            'airspeed_ref', 'altitude_ref', 'course_ref',
                            'airspeed_err', 'altitude_err', 'course_err',
        )

        self.flight_data_logfile: str = flight_data_logfile
        with open(self.flight_data_logfile, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            csv_writer.writeheader()
        
        self.telemetry: Tuple(BoundedProperty, ...) = (
            prp.lat_gc_deg, prp.lng_gc_deg, prp.altitude_sl_ft,
            prp.roll_rad, prp.pitch_rad, prp.heading_rad,
            prp.p_radps, prp.q_radps, prp.r_radps, prp.airspeed_kts,
            prp.throttle_cmd, prp.elevator_cmd, prp.aileron_cmd,
            self.prp_target_airspeed_kts, self.prp_target_pitch_rad, self.prp_target_roll_rad,
            self.prp_airspeed_err, self.prp_pitch_err, self.prp_roll_err
        )


    def reset_task(self, sim: Simulation) -> None:
        # reset task class attributes with initial conditions
        self.reset_target_state(sim)
        sim[self.steps_left] = self.steps_left.max

    def is_terminal(self, sim: Simulation) -> bool:
        # if the episode is done, return True
        is_terminal_step: bool = sim[self.steps_left] <= 0

        # check collision with ground
        is_crashed: bool = sim[prp.altitude_sl_ft] <= 0

        return is_terminal_step or is_crashed

    def get_action_space(self) -> gym.spaces.Box:
        # defining observation space based on pre-chosen state variables
        state_lows: np.ndarray = np.array([state_var.min for state_var in self.state_vars])
        state_highs: np.ndarray = np.array([state_var.max for state_var in self.state_vars])
        observation_space = gym.spaces.Box(low=state_lows, high=state_highs, dtype=np.float32)
        return observation_space


    def get_observation_space(self) -> gym.spaces.Box:
        # define action space
        action_lows: np.ndarray = np.array([action_var.min for action_var in self.action_vars])
        action_highs: np.ndarray = np.array([action_var.max for action_var in self.action_vars])
        action_space = gym.spaces.Box(low=action_lows, high=action_highs, dtype=np.float32)
        return action_space


    def observe_state(self, sim: Simulation, first_obs: bool = False) -> np.ndarray:
        # create state named tuple with state variable values from the sim properties
        self.update_errors(sim)
        state: self.State = self.State(*[sim[prop] for prop in self.state_vars])
        state_nparray: np.ndarray = np.array(state)
        # return state as a numpy array
        return state_nparray


    def update_errors(self, sim: Simulation) -> None:
        sim[self.prp_airspeed_err] = sim[self.prp_target_airspeed_kts] - sim[prp.airspeed_kts]
        sim[self.prp_roll_err] = sim[self.prp_target_roll_rad] - sim[prp.roll_rad]
        sim[self.prp_pitch_err] = sim[self.prp_target_pitch_rad] - sim[prp.pitch_rad]


    def set_target_state(self, sim: Simulation, target_airspeed_kts: float, target_roll_rad: float, target_pitch_rad: float) -> None:
        # set target state namedtuple with target state attributes
        self.target = self.TargetState(str(target_airspeed_kts), str(target_roll_rad), str(target_pitch_rad))
        sim[self.prp_target_airspeed_kts] = target_airspeed_kts
        sim[self.prp_target_roll_rad] = target_roll_rad
        sim[self.prp_target_pitch_rad] = target_pitch_rad


    def reset_target_state(self, sim: Simulation):
        # reset task class attributes with initial conditions
        sim[self.prp_target_airspeed_kts]: float = sim[prp.initial_airspeed_kts]
        sim[self.prp_target_roll_rad]: float = sim[prp.initial_roll_rad]
        sim[self.prp_target_pitch_rad]: float = sim[prp.initial_pitch_rad]

    def flight_data_logging(self, sim: Simulation):
       # write flight data to csv
        with open(self.flight_data_logfile, 'a') as csv_file:
            csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            info: dict[str, float] = {}
            for fieldname, prop in zip(self.fieldnames, self.telemetry):
                info[fieldname] = sim[prop]
            csv_writer.writerow(info)
