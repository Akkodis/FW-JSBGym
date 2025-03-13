import gymnasium as gym
import numpy as np

from typing import Deque
from collections import deque
from omegaconf import DictConfig
from fw_jsbgym.envs.jsbsim_env import JSBSimEnv
from abc import ABC

class JSBSimTask(JSBSimEnv, ABC):
    """
    Abstract class for JSBSim tasks.
    """
    def __init__(self, cfg_env: DictConfig, telemetry_file: str='', render_mode:str='none') -> None:
        """
            Args:
                - `cfg_env`: DictConfig object containing the environment configuration
                - `telemetry_file`: the name of the file containing the flight data to be logged
                - `render_mode`: the render mode for the task
        """
        super().__init__(cfg_env, telemetry_file, render_mode)
        self.task_cfg: DictConfig = cfg_env.task

        # declaring observation. Deque with a maximum length of obs_history_size
        self.observation_deque: Deque[np.ndarray] = deque(maxlen=self.task_cfg.mdp.obs_hist_size) # deque of 1D nparrays containing self.State

        # declaring action history. Deque with a maximum length of act_history_size
        self.action_hist: Deque[np.ndarray] = deque(maxlen=self.task_cfg.mdp.act_hist_size) # action type: np.ndarray


    def reset_props(self) -> None:
        """
            Reset some of the properties (target, errors, action history, action averages, fcs position history etc...)
        """
        super().reset_props() # reset the parent class JSBSimEnv properties

        self.update_action_history() # reset action history


    def update_action_history(self, action: np.ndarray=None) -> None:
        """
            Update the action history with the newest action and drop the oldest action.
            If it's the first action, the action history is initialized to `obs_history_size` * `action`.
        """
        # if it's the first action -> action is None: fill action history with zeros
        init_action: np.ndarray = np.zeros(self.action_space.shape, dtype=np.float32)
        if action is None:
            for _ in range(self.task_cfg.mdp.obs_hist_size):
                self.action_hist.append(init_action)
        # else just append the newest action
        else:
            self.action_hist.append(action)


    def observe_state(self, first_obs: bool = False) -> np.ndarray:
        """
            Observe the state of the aircraft, i.e. the state variables defined in the `state_vars` tuple, `obs_history_size` times.\\
            If it's the first observation, the observation is initialized to `obs_history_size` * `state`.\\
            Otherwise the observation is the newest `state` appended to the observation history and the oldest is dropped.
        """
        # observe the state of the aircraft, self.state gets updated here
        state = super().observe_state()

        # if it's the first observation i.e. following a reset(): fill observation with obs_history_size * state
        if first_obs:
            for _ in range(self.task_cfg.mdp.obs_hist_size):
                self.observation_deque.append(state)
        # else just append the newest state
        else:
            self.observation_deque.append(state)

        # return observation as a numpy array and add one channel dim for CNN policy
        if self.task_cfg.mdp.obs_is_matrix:
            obs: np.ndarray = np.expand_dims(np.array(self.observation_deque), axis=0).astype(np.float32)
        else: # else return observation as a vector for MLP policy
            obs: np.ndarray = np.array(self.observation_deque).squeeze().flatten().astype(np.float32)
        return obs

