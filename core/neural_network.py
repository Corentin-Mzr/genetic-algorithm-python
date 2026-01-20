from abc import ABC, abstractmethod

import numpy as np

class NeuralNetwork(ABC):
    fitness: float = 0.0
    
    @abstractmethod
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        """ From environment state returns the index of the action to do """
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """ Returns flattened weights """
        pass
    
    @abstractmethod
    def set_weights(self, weights: np.ndarray) -> np.ndarray:
        """ Set weights from flattened array """
        pass
