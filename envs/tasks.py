import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from simulation.jsb_simulation import Simulation
from typing import Tuple, Dict

class Task(ABC):
    @abstractmethod
    def task_step(self, sim: Simulation, action: np.ndarray, sim_steps_after_agent_action: int) -> Tuple[np.ndarray, float, bool: Dict]:


class FlightTask(ABC, Task):
    