import gymnasium as gym
import numpy as np
import random
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

        self.fcs_pos_hist = []

        self.sim_options: dict = {}


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
        # super().reset(seed=seed)
        # if seed is not None:
        #     self.sim["simulation/randomseed"] = seed
        # else:
        #     self.sim["simulation/randomseed"] = np.random.randint(0, 10000)

        # set render mode
        if options is not None:
            if "render_mode" in options: 
                self.render_mode = options["render_mode"]
            # Get options dict and store in as an attribute to keep it across resets
            # (the options argument is set to None when SyncVectorEnv autoresets the envs)
            # setup wind and turbulence
            if "seed" in options:
                self.sim_options["seed"] = options["seed"]
            if "atmosphere" in options:
                self.sim_options["atmosphere"] = options["atmosphere"]

        # TODO add curriculum learning with a bool in args.config yaml file and change
        # the sim_options dict accordingly
        if self.sim_options is not None:
            if "seed" in self.sim_options:
                self.sim["simulation/randomseed"] = self.sim_options["seed"]
            else:
                self.sim["simulation/randomseed"] = np.random.randint(0, 10000)
        print(f"Seed: {self.sim['simulation/randomseed']}")

        # set the atmospehere (wind and turbulences)
        self.set_atmosphere(self.sim_options["atmosphere"])

    def set_atmosphere(self, atmo_options: dict={}) -> None:
        """
            Set the atmosphere (wind and turbulences) of the environment.
        """
        # set default wind and turb values
        wspeed_n, wspeed_e, wspeed_d = 0.0, 0.0, 0.0
        turb_type, turb_w20_fps, turb_severity, severity = 3, 0, 0, 0
        severity_options = ["off", "light", "moderate", "severe"]
        wind_vec = np.zeros(3)
        if len(atmo_options) != 0:
            if atmo_options.get("variable", False): # random wind and turbulence magnitudes
                severity = random.choice(severity_options)
                print(f"Variable Severity")
            else: # fixed wind and turbulence magnitudes
                severity = atmo_options.get("severity", None)
                print(f"Fixed Severity")
            if atmo_options.get("wind", False): # if there's a wind key in dict
                if atmo_options["wind"].get("enable", False): # if wind is enabled
                    if atmo_options["wind"].get("rand_continuous", False): # if continuous random wind
                        wind_vec = self.random_wind_vector(windspeed_limit=82.8)
                        wspeed_n = wind_vec[0] * 0.9115 # kmh to fps
                        wspeed_e = wind_vec[1] * 0.9115 # kmh to fps
                        wspeed_d = wind_vec[2] * 0.9115 # kmh to fps
                    else: # if discrete wind parameters
                        wind_dir = self.random_wind_direction()
                        match severity:
                            case "off": # no wind
                                wspeed_n= 0.0
                                wspeed_e = 0.0
                                wspeed_d = 0.0
                                print("No Wind")
                            case "light": # light wind
                                wind_vec = wind_dir * 25.2 # 7 mps = 25.2 kph
                                wspeed_n = wind_vec[0] * 0.9115 # kph to fps
                                wspeed_e = wind_vec[1] * 0.9115 # kph to fps
                                wspeed_d = wind_vec[2] * 0.9115 # kph to fps
                                print("Light Wind")
                            case "moderate": # moderate wind
                                wind_vec = wind_dir * 54 # 15 mps = 54 kph
                                wspeed_n = wind_vec[0] * 0.9115 # kph to fps
                                wspeed_e = wind_vec[1] * 0.9115 # kph to fps
                                wspeed_d = wind_vec[2] * 0.9115 # kph to fps
                                print("Moderate Wind")
                            case "severe": # strong wind
                                wind_vec = wind_dir * 82.8 # 23 mps = 82.8 kph
                                wspeed_n = wind_vec[0] * 0.9115 # kph to fps
                                wspeed_e = wind_vec[1] * 0.9115 # kph to fps
                                wspeed_d = wind_vec[2] * 0.9115 # kph to fps
                                print("Severe Wind")
                else: # if wind is disabled
                    wspeed_n = 0.0
                    wspeed_e = 0.0
                    wspeed_d = 0.0
                    print("No Wind")
            else: # if there's no wind key in dict
                wspeed_n = 0.0
                wspeed_e = 0.0
                wspeed_d = 0.0
                print("No Wind")
            if atmo_options["turb"].get("enable", False): # if turbulence is enabled
                match severity:
                    case "off": # no turbulence
                        turb_type = 3
                        turb_w20_fps = 0
                        turb_severity = 0
                        print("No Turbulence")
                    case "light": # light turbulence
                        turb_type = 3
                        turb_w20_fps = 25
                        turb_severity = 3
                        print("Light Turbulence")
                    case "moderate": # moderate turbulence
                        turb_type = 3
                        turb_w20_fps = 50
                        turb_severity = 4
                        print("Moderate Turbulence")
                    case "severe": # severe turbulence
                        turb_type = 3
                        turb_w20_fps = 75
                        turb_severity = 6
                        print("Severe Turbulence")
            else: # if turbulence is disabled
                turb_type = 3
                turb_w20_fps = 0
                turb_severity = 0
                print("No Turbulence")
            if atmo_options["gust"].get("enable", False): # if gust key in dict
                if atmo_options["gust"].get("enable", False):
                    gust_startup_duration_sec = 0.25
                    gust_steady_duration_sec = 0.5
                    gust_end_duration_sec = 0.25
                    gust_frame = 2 # 1: Body frame, 2: Wind frame, 3: inertial NED frame
                    match severity:
                        case "off": # no gust
                            gust_mag_fps = 0
                            print("No Gust")
                        case "light": # light gust
                            gust_mag_fps = 25.2 * 0.9115 # 7 mps = 25.2 kph
                            print("Light Gust")
                        case "moderate": # moderate gust
                            gust_mag_fps = 54 * 0.9115 # 15 mps = 54 kph
                            print("Moderate Gust")
                        case "severe":
                            gust_mag_fps = 82.8 * 0.9115 # 23 mps = 82.8 kph
                            print("Severe Gust")
                    self.sim[prp.gust_startup_duration_sec] = gust_startup_duration_sec
                    self.sim[prp.gust_steady_duration_sec] = gust_steady_duration_sec
                    self.sim[prp.gust_end_duration_sec] = gust_end_duration_sec
                    self.sim[prp.gust_mag_fps] = gust_mag_fps # ft/s
                    self.sim[prp.gust_frame] = gust_frame 

            self.sim[prp.windspeed_north_fps] = wspeed_n
            self.sim[prp.windspeed_east_fps] = wspeed_e
            self.sim[prp.windspeed_down_fps] = wspeed_d
            self.sim[prp.turb_type] = turb_type
            self.sim[prp.turb_w20_fps] = turb_w20_fps
            self.sim[prp.turb_severity] = turb_severity
            print(f"Wind: \n"
                  f"  N: {self.sim[prp.windspeed_north_kph]} kph\n"
                  f"  E: {self.sim[prp.windspeed_east_kph]} kph\n"
                  f"  D: {self.sim[prp.windspeed_down_kph]} kph\n"
                  f" Magnitude: {np.linalg.norm(wind_vec)} kph\n")
        else:
            print(f"WARNING: No Atmosphere Options Found")


    def random_wind_vector(self, windspeed_limit = 82.8):
        wind_dir = self.random_wind_direction()
        wind_norm = np.random.uniform(0, windspeed_limit)
        wind_vec = wind_dir * wind_norm
        return wind_vec


    def random_wind_direction(self):
        rand_vec = np.random.uniform(-1, 1, size=(3))
        unit_vector = rand_vec / np.linalg.norm(rand_vec)
        return unit_vector


    def gust_start(self):
        gust_dir = self.random_wind_direction()
        self.sim[prp.gust_dir_x_fps] = gust_dir[0]
        self.sim[prp.gust_dir_y_fps] = gust_dir[1]
        self.sim[prp.gust_dir_z_fps] = gust_dir[2]
        self.sim[prp.gust_start] = 1
        print("Gust Start")


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
        atmo_options = self.sim_options["atmosphere"]
        if len(atmo_options) != 0:
            if atmo_options["gust"].get("enable"):
                curr_step = self.sim[self.current_step]
                if curr_step == 500 or curr_step == 1500:
                    self.gust_start()


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