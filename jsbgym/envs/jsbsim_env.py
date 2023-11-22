import gymnasium as gym
import numpy as np
import os
import csv
from math import ceil

from typing import Dict

from jsbgym.simulation.jsb_simulation import Simulation
from jsbgym.visualizers.visualizer import PlotVisualizer, FlightGearVisualizer
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty


class JSBSimEnv(gym.Env):
    """
        Gymnasium JSBSim environment for reinforcement learning.
        Attr:
            - `metadata`: metadata of the environment, contains the render modes
            - `sim`: the simulation object containing the JSBSim FDM
            - `task`: the task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
            - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
            - `sim_steps_after_agent_action`: the number of simulation steps to run after the agent action
            - `aircraft_id`: Aircraft to simulate
            - `viz_time_factor`: the factor by which the simulation time is scaled for visualization, only taken into account if render mode is not "none"
            - `plot_viz`: the plot visualizer
            - `fgear_viz`: the FlightGear visualizer
            - `render_mode`: the mode to render the environment, can be one of the following: `["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]`
            - `enable_fgear_output`: whether to enable FlightGear output for JSBSim <-> FGear communcation
            - `action_space`: the action space of the environment
            - `observation_space`: the observation space of the environment
    """
    metadata: Dict[str, str] = {"render_modes": ["none", "log", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]}

    def __init__(self,
                 jsbsim_config: dict,
                 telemetry_cfg: tuple,
                 render_mode: str=None,
                 aircraft_id: str='x8') -> None:

        """
        Gymnasium JSBSim environment for reinforcement learning.

        Args: 
            - `task_type`: the task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
            - `render_mode`: the mode to render the environment, can be one of the following: `["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]`
            - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
            - `agent_frequency`: the frequency of the agent (controller) at which it interacts with the environment
            - `episode_length_s`: the duration of the episode in seconds
            - `aircraft_id`: Aircraft to simulate
            - `viz_time_factor`: the factor by which the simulation time is scaled for visualization, only taken into account if render mode is not "none"
       """
        self.jsbsim_cfg: dict = jsbsim_config

        # simulation attribute, will be initialized in reset() with a call to Simulation()
        self.sim: Simulation = None

        self.episode_length_s: float = self.jsbsim_cfg["episode_length_s"]
        self.agent_frequency: float = self.jsbsim_cfg["agent_freq"]
        self.fdm_frequency: float = self.jsbsim_cfg["fdm_freq"]
        self.sim_steps_after_agent_action: int = int(self.fdm_frequency // self.agent_frequency)
        self.aircraft_id: str = aircraft_id
        self.telemetry_file: str = telemetry_cfg[0]
        self.telemetry_fieldnames: tuple = telemetry_cfg[1]


        # visualizers, one for matplotlib and one for FlightGear
        self.plot_viz: PlotVisualizer = None
        self.fgear_viz: FlightGearVisualizer = None

        # raise error if render mode is not None or not in the render_modes list
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # enable FlightGear output for JSBSim <-> FGear communcation if render mode is fgear, fgear_plot, flear_plot_scale
        self.enable_fgear_output: bool = False
        if self.render_mode in self.metadata["render_modes"][4:]:
            self.enable_fgear_output = True

        # set the visualization time factor (plot and/or flightgear visualization),default is None
        self.viz_time_factor: float = None
        if self.render_mode in self.metadata["render_modes"][2:]:
            self.viz_time_factor: float = jsbsim_config["viz_time_factor"]

        max_episode_steps: int = ceil(self.episode_length_s * self.fdm_frequency)
        self.steps_left: BoundedProperty = BoundedProperty("info/steps_left", "steps remaining in the current episode", 0, max_episode_steps)

        self.reward = None

        if not os.path.exists('telemetry'):
            os.makedirs('telemetry')
        self.telemetry_setup(self.telemetry_file)

    def reset(self, seed: int=None, options: dict=None) -> None:
        """
        Resets the state of the environment and returns an initial observation.

        Args:
            - `seed`: the seed for the random number generator (for gymnasium and JSBSim PNRG)

        Returns:
            - `state`: the initial state of the environment after reset
        """
        if self.sim:
            # reintialize the simulation
            self.sim.fdm.reset_to_initial_conditions(0)
        if self.sim is None:
            # initialize the simulation
            self.sim = Simulation(fdm_frequency =self.fdm_frequency,
                                  aircraft_id=self.aircraft_id,
                                  viz_time_factor=self.viz_time_factor,
                                  enable_fgear_output=self.enable_fgear_output)

        # convert the airspeed from kts to m/s
        self.convert_airspeed_kts2mps()

        # reset the random number generator
        super().reset(seed=seed)
        if seed is not None:
                self.sim["simulation/randomseed"] = seed


    def step(self, action: np.ndarray) -> None:
        """
            Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.

            Args:
                - `action`: an action provided by an agent, to be stepped through the environment's dynamics

            Returns:
                - The `obs` of the environment after the action, the `reward` obtained, whether the episode of terminated - `done`, and additional `info`
        """
        pass


    def render(self) -> None:
        """
            Rendering method. Launches the visualizers according to the render mode.
            The visualizers are launched only once, when the render method is called for the first time.
            This is because the visualizers are one launched as new processes, hence are independent from the main process.
        """
        # launch the visualizers according to the render mode
        if self.render_mode == 'none': pass
        if self.render_mode == 'plot_scale':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(True, self.telemetry_file)
        if self.render_mode == 'plot':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(False, self.telemetry_file)
        if self.render_mode == 'fgear':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
        if self.render_mode == 'fgear_plot':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(False, self.telemetry_file)
        if self.render_mode == 'fgear_plot_scale':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(True, self.telemetry_file)


    def convert_airspeed_kts2mps(self) -> None:
        """
            Converts the airspeed from kts to m/s
        """
        self.sim[prp.airspeed_mps] = self.sim[prp.airspeed_kts] * 0.51444


    def telemetry_setup(self, telemetry_file: str) -> None:
        if telemetry_file is not None: 
            self.telemetry_file = telemetry_file
        if self.render_mode in self.metadata["render_modes"][1:]:
            with open(self.telemetry_file, 'w') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=self.telemetry_fieldnames)
                csv_writer.writeheader()