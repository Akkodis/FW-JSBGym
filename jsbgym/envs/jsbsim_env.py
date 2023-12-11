import gymnasium as gym
import numpy as np
import os
import csv
from math import ceil

from typing import Dict, Tuple, NamedTuple
from collections import namedtuple
from abc import ABC, abstractmethod

from jsbgym.simulation.jsb_simulation import Simulation
from jsbgym.visualizers.visualizer import PlotVisualizer, FlightGearVisualizer
from jsbgym.utils import jsbsim_properties as prp
from jsbgym.utils.jsbsim_properties import BoundedProperty


class JSBSimEnv(gym.Env, ABC):
    """"
        Gymnasium JSBSim environment for reinforcement learning.
        Abstract class. Must be subclassed to implement a task.
        Attr:
            - `metadata`: metadata of the environment, contains the render modes
            - `sim`: the simulation object containing the JSBSim FDM
            - `episode_length_s`: the duration of the episode in seconds
            - `agent_frequency`: the frequency of the agent (controller) at which it interacts with the environment
            - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
            - `sim_steps_after_agent_action`: the number of simulation steps to run after the agent action
            - `aircraft_id`: Aircraft to simulate
            - `telemetry_file`: the path to the telemetry file
            - `telemetry_fieldnames`: the fieldnames of the telemetry file
            - `viz_time_factor`: the factor by which the simulation time is scaled for visualization, only taken into account if render mode is not "none"
            - `plot_viz`: the plot visualizer
            - `fgear_viz`: the FlightGear visualizer
            - `render_mode`: the mode to render the environment, can be one of the following: `["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]`
            - `enable_fgear_output`: whether to enable FlightGear output for JSBSim <-> FGear communcation
            - `action_space`: the action space of the environment
            - `observation_space`: the observation space of the environment
            - `**_prps`: the state, action, target, telemetry and error properties of the environment to be set in task child classes
            - `state`: namedtuple containing the state of the environment, initialized and updated from task child classes
            - `target`: namedtuple containing the target state of the environment, initialized and updated from task child classes
            - `errors`: namedtuple containing the errors of the environment, initialized and updated from task child classes
            - `reward`: the reward of the environment, updated from task child classes
    """
    metadata: Dict[str, str] = {"render_modes": ["none", "log", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]}

    def __init__(self,
                 jsbsim_config: dict,
                 telemetry_file: str,
                 render_mode: str=None,
                 aircraft_id: str='x8') -> None:

        """
        Gymnasium JSBSim environment for reinforcement learning.

        Args: 
            - `render_mode`: the mode to render the environment, can be one of the following: `["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]`
            - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
            - `agent_frequency`: the frequency of the agent (controller) at which it interacts with the environment
            - `episode_length_s`: the duration of the episode in seconds
            - `aircraft_id`: Aircraft to simulate
            - `viz_time_factor`: the factor by which the simulation time is scaled for visualization, only taken into account if render mode is not "none"
       """
        # jsbsim level configuration
        self.jsbsim_cfg: dict = jsbsim_config

        # simulation attribute, will be initialized in reset() with a call to Simulation()
        self.sim: Simulation = None

        # setting up simulation parameters
        self.episode_length_s: float = self.jsbsim_cfg["episode_length_s"]
        self.agent_frequency: float = self.jsbsim_cfg["agent_freq"]
        self.fdm_frequency: float = self.jsbsim_cfg["fdm_freq"]
        self.sim_steps_after_agent_action: int = int(self.fdm_frequency // self.agent_frequency)
        self.aircraft_id: str = aircraft_id
        self.telemetry_file: str = telemetry_file
        self.telemetry_fieldnames: tuple = ()

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

        self.max_episode_steps: int = ceil(self.episode_length_s * self.fdm_frequency)
        self.current_step = BoundedProperty("info/current_step", "current step in the current episode", 0, self.max_episode_steps)
        self.steps_left = BoundedProperty("info/steps_left", "steps remaining in the current episode", 0, self.max_episode_steps)


        ## Generic attributes for an env. Will be set in the task child class.
        # observation of the agent in a numpy array format
        self.observation: np.ndarray = None 

        # Tuples containing all the relevant properties of the env
        self.state_prps: Tuple[BoundedProperty, ...] = ()
        self.action_prps: Tuple[BoundedProperty, ...] = ()
        self.target_prps: Tuple[BoundedProperty, ...] = ()
        self.telemetry_prps: Tuple[BoundedProperty, ...] = ()
        self.error_prps: Tuple[BoundedProperty, ...] = ()

        ## Named tuples containing relevant variables of the env
        # declaring state NamedTuple structure
        self.State: NamedTuple = None
        self.state: self.State = None

        self.TargetState: NamedTuple = None
        self.target: self.TargetState = None

        # declaring error NamedTuple structure
        self.Errors: NamedTuple = None
        self.errors: self.Errors = None

        self.reward: float = None

        self.sim_options: dict = None


    def initialize(self) -> None:
        # initialize state NamedTuple structure
        self.State: NamedTuple = namedtuple('State', [state_prp.get_legal_name() for state_prp in self.state_prps])

        # initialize target state NamedTuple structure
        self.TargetState: NamedTuple = namedtuple('TargetState', [f"target_{t_state_prp.get_legal_name()}" for t_state_prp in self.target_prps])

        # initialize error NamedTuple structure
        self.Errors: NamedTuple = namedtuple('Errors', [f"{error_prp.get_legal_name()}_err" for error_prp in self.error_prps])

        # initialize telemetry fieldnames
        self.telemetry_fieldnames: Tuple[str, ...] = tuple([tele_prp.get_legal_name() for tele_prp in self.telemetry_prps]) 


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

        # reset the random number generator
        super().reset(seed=seed)
        if seed is not None:
            self.sim["simulation/randomseed"] = seed
        else:
            self.sim["simulation/randomseed"] = np.random.randint(0, 10000)

        # set render mode
        if options is not None:
            if "render_mode" in options: 
                self.render_mode = options["render_mode"]
            # Get options dict and store in as an attribute to keep it across resets
            # (the options argument is set to None when SyncVectorEnv autoresets the envs)
            # setup wind and turbulence
            if self.sim_options is None:
                self.sim_options = options

        # TODO add curriculum learning with a bool in args.config yaml file and change
        # the sim_options dict accordingly
        if self.sim_options is not None:
            if "atmosphere" in self.sim_options:
                self.set_atmosphere(self.sim_options["atmosphere"])


    def set_atmosphere(self, atmo_options: dict=None) -> None:
        """
            Set the atmosphere (wind and turbulences) of the environment.
        """
        # set atmosphere
        if atmo_options is not None:
            if atmo_options["rand_magnitudes"]: # random wind and turbulence magnitudes
                if atmo_options["wind"]:
                    wind_vector = self.random_wind_vec(wspeed_limit=90)
                    self.sim[prp.windspeed_north_fps] = wind_vector[0] * 0.9115 # kmh to fps
                    self.sim[prp.windspeed_east_fps] = wind_vector[1] * 0.9115 # kmh to fps
                    self.sim[prp.windspeed_down_fps] = wind_vector[2] * 0.9115 # kmh to fps
                    print(f"Wind: \n"
                            f"  N: {self.sim[prp.windspeed_north_kph]} kph\n" \
                            f"  E: {self.sim[prp.windspeed_east_kph]} kph\n" \
                            f"  D: {self.sim[prp.windspeed_down_kph]} kph\n" \
                            f"  Magnitude: {np.linalg.norm(wind_vector)} kph")
                if atmo_options["turb"]:
                    # turb_severity = np.random.randint(1, 4)
                    turb_severity = np.random.randint(0, 4)
                    match turb_severity:
                        case 0: # no turbulence
                            self.sim[prp.turb_type] = 3
                            self.sim[prp.turb_w20_fps] = 0
                            self.sim[prp.turb_severity] = 0
                            print("No Turbulence")
                        case 1: # light turbulence
                            self.sim[prp.turb_type] = 3
                            self.sim[prp.turb_w20_fps] = 25
                            self.sim[prp.turb_severity] = 3
                            print("Light Turbulence")
                        case 2: # moderate turbulence
                            self.sim[prp.turb_type] = 3
                            self.sim[prp.turb_w20_fps] = 50
                            self.sim[prp.turb_severity] = 4
                            print("Moderate Turbulence")
                        case 3: # severe turbulence
                            self.sim[prp.turb_type] = 3
                            self.sim[prp.turb_w20_fps] = 75
                            self.sim[prp.turb_severity] = 6
                            print("Severe Turbulence")
                else:
                    self.sim[prp.turb_type] = 0
                    self.sim[prp.turb_w20_fps] = 0
                    self.sim[prp.turb_severity] = 0
                    print("No Turbulence") 
            else: # fixed wind and turbulence magnitudes : 58 kmh wind and severe turbulence
                if atmo_options["wind"]:
                    print("Fixed wind : 82 kph N/E")
                    self.sim[prp.windspeed_north_fps] = 58 * 0.9115 # kmh to fps
                    self.sim[prp.windspeed_east_fps] = 58 * 0.9115 # kmh to fps
                if atmo_options["turb"]:
                    print("Fixed turbulence : Severe, W20 = 75 fps")
                    self.sim[prp.turb_type] = 3
                    self.sim[prp.turb_w20_fps] = 75
                    self.sim[prp.turb_severity] = 6


    def random_wind_vec(self, wspeed_limit: int = 30):
        rand_vec = np.random.uniform(-1, 1, size=(3))
        unit_vector = rand_vec / np.linalg.norm(rand_vec)
        wind_norm = np.random.uniform(0, wspeed_limit)
        wind_vector = unit_vector * wind_norm
        return wind_vector


    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """
            Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.

            Args:
                - `action`: an action provided by an agent, to be stepped through the environment's dynamics

            Returns:
                - The `obs` of the environment after the action, the `reward` obtained, whether the episode of terminated - `done`, and additional `info`
        """
        raise NotImplementedError


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


    def get_observation_space(self) -> gym.spaces.Box:
        """
            Get the observation space of the env.
        """
        # defining observation space based on pre-chosen state variables
        state_lows: np.ndarray = np.array([state_var.min for state_var in self.state_prps], dtype=np.float32)
        state_highs: np.ndarray = np.array([state_var.max for state_var in self.state_prps], dtype=np.float32)
        observation_space = gym.spaces.Box(low=np.array(state_lows), high=np.array(state_highs), dtype=np.float32)
        return observation_space


    def get_action_space(self) -> gym.spaces.Box:
        """
            Get the action space of the env.
        """
        # define action space
        action_lows: np.ndarray = np.array([action_var.min for action_var in self.action_prps], dtype=np.float32)
        action_highs: np.ndarray = np.array([action_var.max for action_var in self.action_prps], dtype=np.float32)
        action_space = gym.spaces.Box(low=action_lows, high=action_highs, dtype=np.float32)
        return action_space


    def observe_state(self) -> None:
        """
            Observe the state of the aircraft and update the state properties.
        """
        # update state sim properties
        for state_var in self.state_prps:
            self.sim[state_var] = self.sim[state_var]

        # fill state namedtuple with state variable values from the sim properties
        self.state = self.State(*[self.sim[prop] for prop in self.state_prps])


    @abstractmethod
    def get_reward(self):
        """
            Reward function
        """
        raise NotImplementedError


    def telemetry_logging(self) -> None:
        """
            Log flight data to telemetry csv.
        """
        # write flight data to csv
        with open(self.telemetry_file, 'a') as csv_file:
            csv_writer: csv.DictWriter = csv.DictWriter(csv_file, fieldnames=self.telemetry_fieldnames)
            info: dict[str, float] = {}
            for fieldname, prop in zip(self.telemetry_fieldnames, self.telemetry_prps):
                info[fieldname] = self.sim[prop]
            csv_writer.writerow(info)


    def telemetry_setup(self, telemetry_file: str) -> None:
        """
            Setup the telemetry file and fieldnames.
        """
        if not os.path.exists('telemetry'):
            os.makedirs('telemetry')

        if telemetry_file is not None: 
            self.telemetry_file = telemetry_file
        if self.render_mode in self.metadata["render_modes"][1:]:
            with open(self.telemetry_file, 'w') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=self.telemetry_fieldnames)
                csv_writer.writeheader()