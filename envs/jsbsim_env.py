import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces

from utils import jsbsim_properties as prp
from simulation.jsb_simulation import Simulation
from trim.trim_point import TrimPoint
from typing import Dict, NamedTuple, Type, Tuple
from collections import namedtuple

class JSBSimEnv(gym.Env):
    metadata = {"render_modes": ["plot", "flightgear"]}
    state_vars = (
        prp.altitude_sl_ft, # altitude above mean sea level [ft]
        prp.u_fps, prp.v_fps, prp.w_fps, # body frame velocities [ft/s]
        prp.p_radps, prp.q_radps, prp.r_radps, # body frame angular rates [rad/s]
        prp.airspeed, # true airspeed [ft/s]
        prp.elevator, prp.aileron_left, prp.aileron_right, # control surface positions normalized [-1, 1]
        prp.throttle # throttle position normalized [0, 1]
        )
    action_vars = (
            prp.elevator_cmd, prp.aileron_cmd, # control surface commands normalized [-1, 1]
            prp.throttle_cmd # throttle command normalized [0, 1]
        )
    State: Type[NamedTuple]
    TargetState: Type[NamedTuple]
    def __init__(self, 
                 render_mode=None,
                 fdm_frequency=120.0,
                 agent_frequency=60.0,
                 episode_time_s=60.0,
                 aircraft_id='x8',
                 viz_time_factor=1.0,
                 enable_fgear_viz=False,
                 enable_trim=False,
                 trim_point=None) -> None:
        # simulation attribute, will be initialized in reset() with a call to Simulation()
        self.sim: Simulation = None
        self.fdm_frequency: float = fdm_frequency
        self.sim_steps_after_agent_action: int = self.fdm_frequency // agent_frequency
        self.aircraft_id: str = aircraft_id
        self.viz_time_factor: float = viz_time_factor
        self.enable_fgear_viz: bool = enable_fgear_viz
        self.enable_trim: bool = enable_trim
        self.trim_point: TrimPoint = trim_point
        self.episode_steps: int = math.ceil(episode_time_s * fdm_frequency)
        self.steps_left: prp.BoundedProperty = prp.BoundedProperty("info/steps_left", "steps remaining in the current episode", 0, self.episode_steps)

        # raise error if render mode is not None or not in the render_modes list
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode: render_mode

        # defining observation space based on pre-chosen state variables !!! Add errors a
        state_lows: np.array = np.array([state_var.min for state_var in self.state_vars])
        state_highs: np.array = np.array([state_var.max for state_var in self.state_vars])
        self.observation_space = spaces.Box(low=state_lows, high=state_highs, dtype=np.float32)

        # define action space
        action_lows = np.array([action_var.min for action_var in self.action_vars])
        action_highs = np.array([action_var.max for action_var in self.action_vars])
        self.action_space = spaces.Box(low=action_lows, high=action_highs, dtype=np.float32)

        # create state named tuple
        self.State = namedtuple('State', [state_var.get_legal_name() for state_var in self.state_vars])

        # create target state named tuple
        self.TargetState = namedtuple('TargetState', [f"target_{state_var.get_legal_name()}" for state_var in self.state_vars])
        self.target: self.TargetState = None


    def reset(self) -> np.ndarray:
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """
        if self.sim:
            # reintialize the simulation
            self.sim.fdm.reset_to_initial_conditions(0)
            pass
        if self.sim is None:
            # initialize the simulation
            self.sim = Simulation(fdm_frequency =self.fdm_frequency,
                                  aircraft_id=self.aircraft_id,
                                  viz_time_factor=self.viz_time_factor,
                                  enable_trim=self.enable_trim,
                                  trim_point=self.trim_point)
        state: np.ndarray = self.observe_state()
        return state


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")
        for prop, command in zip(self.action_vars, action):
            self.sim.fdm[prop.name] = command

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()

        # decrement the steps left
        self.sim.fdm[self.steps_left.name] -= 1
        
        # get the state
        state: np.ndarray = self.observe_state()

        # get the reward
        reward: float = self.reward()

        # check if the episode is done
        done = self.is_terminal()

        info: Dict = {"steps_left": self.sim.fdm[self.steps_left.name],
                      "reward": reward}

        return state, reward, done, info



    def observe_state(self) -> np.ndarray:
        return np.array(self.State(*(self.sim.fdm[prop.name] for prop in self.state_vars)))

    def reward(self, state: Type[NamedTuple]) -> float:

        pass

    def is_terminal(self) -> bool:
        # if the episode is done, return True
        is_terminal_step: bool = self.sim.fdm[self.steps_left.name] <= 0

        # check collision with ground
        is_crashed: bool = self.sim.fdm[prp.altitude_sl_ft.name] <= 0

        return is_terminal_step or is_crashed


    def set_target(self, t_altitude, t_u_fps, t_airspeed) -> None:
        """
        Sets the target state for the UAV to reach
        """
        self.target = self.TargetState(prp.)
        pass
