import gymnasium as gym
import numpy as np

from envs.tasks import AttitudeControlTask
from simulation.jsb_simulation import Simulation
from typing import Dict, Type, Tuple
from visualizers.visualizer import PlotVisualizer, FlightGearVisualizer


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
    metadata: Dict[str, str] = {"render_modes": ["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]}

    def __init__(self,
                 task_type: Type[AttitudeControlTask],
                 render_mode: str=None,
                 fdm_frequency: float=240.0, # 120Hz being the default frequency of the JSBSim FDM
                 agent_frequency: float=60.0,
                 episode_time_s: float=60.0,
                 aircraft_id: str='x8',
                 viz_time_factor: float=1.0,
                 obs_history_size: int=5) -> None:

        """
        Gymnasium JSBSim environment for reinforcement learning.

        Args: 
            - `task_type`: the task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
            - `render_mode`: the mode to render the environment, can be one of the following: `["none", "plot", "plot_scale", "fgear", "fgear_plot", "fgear_plot_scale"]`
            - `fdm_frequency`: the frequency of the flight dynamics model (JSBSim) simulation
            - `agent_frequency`: the frequency of the agent (controller) at which it interacts with the environment
            - `episode_time_s`: the duration of the episode in seconds
            - `aircraft_id`: Aircraft to simulate
            - `viz_time_factor`: the factor by which the simulation time is scaled for visualization, only taken into account if render mode is not "none"
       """

        # simulation attribute, will be initialized in reset() with a call to Simulation()
        self.sim: Simulation = None
        # task to perform, implemented as a wrapper, customizing action, observation, reward, etc. according to the task
        self.task = task_type(fdm_freq=fdm_frequency,
                              flight_data_logfile="data/gym_flight_data.csv",
                              episode_time_s=episode_time_s,
                              obs_history_size=obs_history_size)
        self.fdm_frequency: float = fdm_frequency
        self.sim_steps_after_agent_action: int = int(self.fdm_frequency // agent_frequency)
        self.aircraft_id: str = aircraft_id


        # visualizers, one for matplotlib and one for FlightGear
        self.plot_viz: PlotVisualizer = None
        self.fgear_viz: FlightGearVisualizer = None

        # raise error if render mode is not None or not in the render_modes list
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # enable FlightGear output for JSBSim <-> FGear communcation if render mode is fgear, fgear_plot, flear_plot_scale
        self.enable_fgear_output: bool = False
        if self.render_mode in self.metadata["render_modes"][3:]:
            self.enable_fgear_output = True

        # set the visualization time factor (plot and/or flightgear visualization),default is None
        self.viz_time_factor: float = None
        if self.render_mode in self.metadata["render_modes"][1:]:
            self.viz_time_factor: float = viz_time_factor

        # get action and observation space from the task
        self.action_space = self.task.get_action_space()
        self.observation_space = self.task.get_observation_space()


    def reset(self, seed=None) -> np.ndarray:
        """
        Resets the state of the environment and returns an initial observation.

        Args:
            - `seed`: the seed for the random number generator (for gymnasium and JSBSim PNRG)

        Returns:
            - `state`: the initial state of the environment after reset
        """
        # reset the random number generator
        super().reset(seed=seed)
        if seed is not None:
                self.sim["simulation/randomseed"] = seed

        if self.sim:
            # reintialize the simulation
            self.sim.fdm.reset_to_initial_conditions(0)
        if self.sim is None:
            # initialize the simulation
            self.sim = Simulation(fdm_frequency =self.fdm_frequency,
                                  aircraft_id=self.aircraft_id,
                                  viz_time_factor=self.viz_time_factor,
                                  enable_fgear_output=self.enable_fgear_output)
        # reset the task
        obs: np.ndarray = self.task.reset_task(self.sim)

        # launch the environment visualizers
        self.render()

        return obs


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
            Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.

            Args:
                - `action`: an action provided by an agent, to be stepped through the environment's dynamics

            Returns:
                - The `obs` of the environment after the action, the `reward` obtained, whether the episode of terminated - `done`, and additional `info`
        """
        # check if the action is valid
        if action.shape != self.action_space.shape:
            raise ValueError("Action shape is not valid.")

        # apply the action to the simulation
        for prop, command in zip(self.task.action_vars, action):
            self.sim[prop] = command

        # run the simulation for sim_steps_after_agent_action steps
        for _ in range(self.sim_steps_after_agent_action):
            self.sim.run_step()
            # write the telemetry to a log csv file every fdm step (as opposed to every agent step -> to put out of this for loop)
            self.task.flight_data_logging(self.sim)
            # decrement the steps left
            self.sim[self.task.steps_left] -= 1

        # update the errors
        self.task.update_errors(self.sim)

        # get the state
        state: np.ndarray = self.task.observe_state(self.sim)

        # get the reward
        reward: float = self.task.reward(self.sim)

        # check if the episode is done
        done: bool = self.task.is_terminal(self.sim)

        # info dict for debugging and misc infos
        info: Dict = {"steps_left": self.sim[self.task.steps_left],
                      "reward": reward}

        return state, reward, done, info


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
                self.plot_viz = PlotVisualizer(scale=True)
        if self.render_mode == 'plot':
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=False)
        if self.render_mode == 'fgear':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
        if self.render_mode == 'fgear_plot':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=False)
        if self.render_mode == 'fgear_plot_scale':
            if not self.fgear_viz:
                self.fgear_viz = FlightGearVisualizer(self.sim)
            if not self.plot_viz:
                self.plot_viz = PlotVisualizer(scale=True)


