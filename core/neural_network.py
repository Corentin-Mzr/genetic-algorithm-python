from abc import ABC, abstractmethod

import numpy as np

from core.types import Shape

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

def weight_init_uniform(shape: Shape, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """ Weight initializer with uniform distribution """
    return np.random.uniform(a, b, size=shape)

def weight_init_normal(shape: Shape, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """ Weight initializer with normal distribution """
    return np.random.normal(mean, std, size=shape)

def weight_init_lecun(shape: Shape) -> np.ndarray:
    """ Weight initializer LeCun """
    fan_in, fan_out = shape
    limit = np.sqrt(3.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)

def weight_init_he(shape: Shape) -> np.ndarray:
    """ Weight initializer when using ReLU as activation function """
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0.0, std, size=shape)

def weight_init_xavier(shape: Shape) -> np.ndarray:
    """ Weight initializer when using sigmoid as activation function """
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def weight_init_orthogonal(shape: Shape) -> np.ndarray:
    """ Weight initialize when using tanh as activation function """
    # fan_in, fan_out = shape
    # flat_shape = (fan_in, fan_out)
    
    a = np.random.normal(0.0, 1.0, shape)
    u, v = np.linalg.qr(a)
    q = u if u.shape == shape else v
    
    return q * np.sqrt(2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ Sigmoid activation function """
    return 1 / (1 + np.exp(-x))
        
def tanh(x: np.ndarray) -> np.ndarray:
    """ Tanh activation function """
    return np.tanh(x)
    
def relu(x: np.ndarray) -> np.ndarray:
    """ ReLU activation function """
    return np.maximum(0, x)
    
def softmax(x: np.ndarray) -> np.ndarray:
    """ Softmax activation function """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)