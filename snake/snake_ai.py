import numpy as np

from constants import Shape
from core.neural_network import NeuralNetwork

def weight_init_he(shape: Shape) -> np.ndarray:
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape)

def weight_init_xavier(shape: Shape) -> np.ndarray:
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

class SnakeAI(NeuralNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.w1 = weight_init_xavier((input_size, hidden_size))
        self.b1 = np.zeros((hidden_size))
        
        self.w2 = weight_init_xavier((hidden_size, hidden_size))
        self.b2 = np.zeros((hidden_size))
        
        self.w3 = weight_init_xavier((hidden_size, output_size))
        self.b3 = np.zeros((output_size))
        
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        # First layer
        x = np.dot(input, self.w1) + self.b1
        x = self.activate(x)
        
        # Second layer
        x = np.dot(x, self.w2) + self.b2
        x = self.activate(x)
        
        # Output
        x = np.dot(x, self.w3) + self.b3
        return x
    
    def get_action(self, state: np.ndarray) -> int:
        output = self.forward(state)
        return int(np.argmax(output))
    
    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten(),
            self.w3.flatten(),
            self.b3.flatten(),
        ])
        
    def set_weights(self, weights: np.ndarray) -> None:
        idx = 0
        
        w1_size = self.w1.size
        self.w1 = weights[idx:idx+w1_size].reshape(self.w1.shape)
        idx += w1_size
        
        b1_size = self.b1.size
        self.b1 = weights[idx:idx+b1_size]
        idx += b1_size
        
        w2_size = self.w2.size
        self.w2 = weights[idx:idx+w2_size].reshape(self.w2.shape)
        idx += w2_size
        
        b2_size = self.b2.size
        self.b2 = weights[idx:idx+b2_size]
        idx += b2_size
        
        w3_size = self.w3.size
        self.w3 = weights[idx:idx+w3_size].reshape(self.w3.shape)
        idx += w3_size
        
        self.b3 = weights[idx:]

    