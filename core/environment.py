from abc import ABC, abstractmethod

import numpy as np

Observation = tuple[np.ndarray, float, bool]

class Environment(ABC):
    
    @property
    @abstractmethod
    def input_size(self) -> int:
        """ Number of input neurons """
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """ Number of available actions """
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """ Reset the environment back to the initial state """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Observation:
        """ Do one step, returns (new_state, reward, terminated) """
        pass
    
    