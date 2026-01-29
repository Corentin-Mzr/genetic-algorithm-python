from core.neural_network import NeuralNetwork

import numpy as np

from game_2048.constants import Shape

def weight_init_he(shape: Shape) -> np.ndarray:
    """ When using ReLU """
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape)

def weight_init_xavier(shape: Shape) -> np.ndarray:
    """ When using sigmoid """
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def weight_init_orthogonal(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """ When using tanh """
    fan_in, fan_out = shape
    
    flat_shape = (fan_in, fan_out)
    a = np.random.normal(0.0, 1.0, flat_shape)
    
    # QR decomposition to make an orthogonal matrix
    u, v = np.linalg.qr(a)
    
    # q uniform
    q = u if u.shape == flat_shape else v
    
    # Apply gain, for tanh q = sqrt(2)
    return q * gain

class Game2048AI(NeuralNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        gain_tanh = np.sqrt(2)
        
        self.w1 = weight_init_orthogonal((input_size, hidden_size), gain=gain_tanh)
        self.b1 = np.zeros((hidden_size))
        
        self.w2 = weight_init_orthogonal((hidden_size, output_size), gain=gain_tanh)
        self.b2 = np.zeros((output_size))
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        # First layer
        x = np.dot(input, self.w1) + self.b1
        x = self.tanh(x)
        
        # Output
        x = np.dot(x, self.w2) + self.b2
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
        
        self.b2 = weights[idx:]
        
        
if __name__ == '__main__':
    ai = Game2048AI(16, 8, 4)
    weights = ai.get_weights()
    print(weights)