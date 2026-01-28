from typing import Callable

import numpy as np

from core.neural_network import NeuralNetwork

Observation = tuple[np.ndarray, float, bool]
NNFactoryFunc = Callable[[], NeuralNetwork]