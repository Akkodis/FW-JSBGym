import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.tasks import AttitudeControlTask
from simulation.jsb_simulation import Simulation
from typing import Dict, NamedTuple, Type, Tuple
from visualizers.visualizer import PlotVisualizer, FlightGearVisualizer


class JSBSimEnv(gym.Env):
    metadata = {"render_modes": ["plot", "plot_scale", "fgear", "fgear_plt", "fgear_plt_scale"]}

    def __init__(self,
                 task_type: Type[AttitudeControlTask],
                 render_mode=None,
                 fdm_frequency=120.0,
                 agent_frequency=60.0,
                 episode_time_s=60.0,
                 aircraft_id='x8',
                 viz_time_factor=1.0) -> None:

        # simulation attribute, will be initialized in reset() with a call to Simulation()
        self.sim: Simulation = None
        # task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
        self.task = task_type(aircraft_id=aircraft_id,
                              fdm_freq=fdm_frequency,
                              flight_data_logfile="data/gym_flight_data.csv",
                              episode_time_s=episode_time_s)
        self.fdm_frequency: float = fdm_frequency
        self.sim_steps_after_agent_action: int = int(self.fdm_frequency // agent_frequency)
        self.aircraft_id: str = aircraft_id
        self.viz_time_factor: float = viz_time_factor
        self.plot_viz: PlotVisualizer = None
        self.fgear_viz: FlightGearVisualizer = None

        # raise error if render mode is not None or not in the render_modes list
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.enable_fgear_output: bool = False
        if self.render_mode in self.metadata["render_modes"][2:]:
            self.enable_fgear_output = True
        print(self.metadata["render_modes"][:-3])
        print('enable fgear = ', self.enable_fgear_output)
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
                                  enable_fgear_output=self.enable_fgear_output)
        # reset the task
        self.task.reset_task(self.sim)

        # observe the first state after reset and return it
        state: np.ndarray = self.task.observe_state(self.sim)

        self.render()

        return state


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")
        for prop, command in zip(self.task.action_vars, action):
            self.sim[prop] = command

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()

        # decrement the steps left
        self.sim[self.task.steps_left] -= 1
        self.task.update_errors(self.sim)

        # get the state
        state: np.ndarray = self.task.observe_state(self.sim)

        # get the reward
        reward = 0
        # reward: float = self.reward()

        # check if the episode is done
        done = self.task.is_terminal(self.sim)

        # write the telemetry to a log csv file
        self.task.flight_data_logging(self.sim)

        info: Dict = {"steps_left": self.sim[self.task.steps_left],
                      "reward": reward}
        # self.render()

        return state, reward, done, info


    def render(self) -> None:
        if self.render_mode == 'plot_scale':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=True)
        if self.render_mode == 'plot':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=False)
        if self.render_mode == 'fgear':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
        if self.render_mode == 'fgear_plt':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=False)
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
        if self.render_mode == 'fgear_plt_scale':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=True)
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)

