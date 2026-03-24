import numpy as np

from core.neural_network import NeuralNetwork, weight_init_he, relu


class SteeringAI(NeuralNetwork):
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

    def forward(self, input: np.ndarray) -> np.ndarray:
        # First layer
        x = np.dot(input, self.w1) + self.b1
        x = relu(x)

        # Second layer
        x = np.dot(x, self.w2) + self.b2
        x = relu(x)

        # Output
        x = np.dot(x, self.w3) + self.b3
        return x

    def get_action(self, state: np.ndarray) -> int:
        output = self.forward(state)
        return int(np.argmax(output))

    def get_weights(self) -> np.ndarray:
        return np.concatenate(
            [
                self.w1.flatten(),
                self.b1.flatten(),
                self.w2.flatten(),
                self.b2.flatten(),
                self.w3.flatten(),
                self.b3.flatten(),
            ]
        )

    def set_weights(self, weights: np.ndarray) -> None:
        idx = 0

        w1_size = self.w1.size
        self.w1 = weights[idx : idx + w1_size].reshape(self.w1.shape).copy()
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = weights[idx : idx + b1_size].copy()
        idx += b1_size

        w2_size = self.w2.size
        self.w2 = weights[idx : idx + w2_size].reshape(self.w2.shape).copy()
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = weights[idx : idx + b2_size].copy()
        idx += b2_size

        w3_size = self.w3.size
        self.w3 = weights[idx : idx + w3_size].reshape(self.w3.shape).copy()
        idx += w3_size

        b3_size = self.b3.size
        self.b3 = weights[idx : idx + b3_size].copy()


if __name__ == "__main__":
    from steering.constants import AGENT_RAY_COUNT

    ai = SteeringAI(AGENT_RAY_COUNT + 1, 8, 3)
    weights = ai.get_weights()
    # print(weights)

    inp = np.array([-1, -1, -1, -1, -1, -1, 0.8])

    state = np.array(inp)
    r = ai.forward(state)
    print(r)
    action = ai.get_action(state)
    print(action)
