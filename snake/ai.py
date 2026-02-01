import numpy as np

from core.neural_network import NeuralNetwork, weight_init_he, relu, softmax

class SnakeAI(NeuralNetwork):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.w1 = weight_init_he((input_size, hidden_size))
        self.b1 = np.zeros((hidden_size))
        
        self.w2 = weight_init_he((hidden_size, hidden_size))
        self.b2 = np.zeros((hidden_size))
        
        self.w3 = weight_init_he((hidden_size, output_size))
        self.b3 = np.zeros((output_size))
        
        self.activations: dict[str, np.ndarray] = {
            'input': np.zeros((input_size,)),
            'hidden1': np.zeros((hidden_size,)),
            'hidden2': np.zeros((hidden_size,)),
            'output': np.zeros((output_size,)),
        }
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.activations['input'] = input
        
        # First layer
        x = np.dot(input, self.w1) + self.b1
        x = relu(x)
        self.activations['hidden1'] = x
        
        # Second layer
        x = np.dot(x, self.w2) + self.b2
        x = relu(x)
        self.activations['hidden2'] = x
        
        # Output
        x = np.dot(x, self.w3) + self.b3
        x = softmax(x)
        self.activations['output'] = x
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

if __name__ == '__main__':
    ai = SnakeAI(4, 7, 3)
    weights = ai.get_weights()
    print(weights)
    print(ai.forward(np.array([0.1, 0.2, 0.3, 0.5])))