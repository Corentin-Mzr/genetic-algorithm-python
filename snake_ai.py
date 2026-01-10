import numpy as np
from snake_game import RELATIVE_ACTIONS, DirectionRelative
from constants import Shape

def weight_init_he(shape: Shape) -> np.ndarray:
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape)

def weight_init_xavier(shape: Shape) -> np.ndarray:
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

class SnakeAI:
    def __init__(self, input_size: int = 18, hidden_size: int = 9, output_size: int = 3):
        self.w1: np.ndarray = weight_init_xavier((input_size, hidden_size))
        self.b1: np.ndarray = np.zeros((hidden_size))
        self.w2: np.ndarray = weight_init_xavier((hidden_size, output_size))
        self.b2: np.ndarray = np.zeros((output_size))
        # self.w3: np.ndarray = np.random.uniform(-1, 1, size=(hidden_size, output_size))
        # self.b3: np.ndarray = np.zeros((output_size))
        
        self.fitness: float = 0.0
        self.raw_score: float = 0.0
        
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        x: np.ndarray = np.dot(input, self.w1) + self.b1
        x = self.activate(x)
        x = np.dot(x, self.w2) + self.b2
        # x = self.activate(x)
        # x = np.dot(x, self.w3) + self.b3
        return x
    
    def get_action(self, state: np.ndarray) -> DirectionRelative:
        output: np.ndarray = self.forward(state)
        action: np.intp = np.argmax(output)
        return RELATIVE_ACTIONS[action]
    
    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten(),
            # self.w3.flatten(),
            # self.b3.flatten(),
        ])
        
    def set_weights(self, weights: np.ndarray) -> None:
        idx: int = 0
        
        w1_size: int = self.w1.size
        self.w1 = weights[idx:idx+w1_size].reshape(self.w1.shape)
        idx += w1_size
        
        b1_size: int = self.b1.size
        self.b1 = weights[idx:idx+b1_size]
        idx += b1_size
        
        w2_size: int = self.w2.size
        self.w2 = weights[idx:idx+w2_size].reshape(self.w2.shape)
        idx += w2_size
        
        self.b2 = weights[idx:]
        
        # b2_size: int = self.b2.size
        # self.b2 = weights[idx:idx+b2_size]
        # idx += b2_size
        
        # w3_size: int = self.w3.size
        # self.w3 = weights[idx:idx+w3_size].reshape(self.w3.shape)
        # idx += w3_size
        
        # self.b3 = weights[idx:]


if __name__ == '__main__':
    ai = SnakeAI()
    print(ai.w1)
    print(ai.b1)
    print(ai.w2)
    print(ai.b2)
    # print(ai.w3)
    # print(ai.b3)
    