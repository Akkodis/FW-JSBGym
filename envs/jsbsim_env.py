import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.tasks import AttitudeControlTask, Task
from simulation.jsb_simulation import Simulation
from trim.trim_point import TrimPoint
from typing import Dict, NamedTuple, Type, Tuple
from collections import namedtuple

class JSBSimEnv(gym.Env):
    metadata = {"render_modes": ["plot", "flightgear"]}

    def __init__(self,
                 task_type: Type[Task],
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
        # task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
        self.task = task_type(aircraft_id, episode_time_s, fdm_frequency)
        self.fdm_frequency: float = fdm_frequency
        self.sim_steps_after_agent_action: int = self.fdm_frequency // agent_frequency
        self.aircraft_id: str = aircraft_id
        self.viz_time_factor: float = viz_time_factor
        self.enable_fgear_viz: bool = enable_fgear_viz
        self.enable_trim: bool = enable_trim
        self.trim_point: TrimPoint = trim_point

        # raise error if render mode is not None or not in the render_modes list
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode: render_mode

        self.action_space = self.task.get_action_space()
        self.observation_space = self.task.get_observation_space()


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
        # reset the task's target
        self.task.reset_target_state(self.sim)

        # observe the first state after reset and return it
        state: np.ndarray = self.task.observe_state(self.sim)
        return state


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")
        for prop, command in zip(self.action_vars, action):
            self.sim[prop] = command

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()

        # decrement the steps left
        self.task.sim[self.steps_left] -= 1
        
        # get the state
        state: np.ndarray = self.task.observe_state(self.sim)

        # get the reward
        reward = 0
        # reward: float = self.reward()

        # check if the episode is done
        done = self.task.is_terminal()

        info: Dict = {"steps_left": self.sim.fdm[self.steps_left.name],
                      "reward": reward}

        return state, reward, done, info

