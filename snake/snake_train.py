from snake_ai import SnakeAI
from snake_wrapper import SnakeGameWrapper
from core.train import train

if __name__ == '__main__':
    def snake_factory() -> SnakeAI:
        return SnakeAI(input_size=SnakeGameWrapper().input_size, hidden_size=7, output_size=SnakeGameWrapper().output_size)
    
    train(SnakeGameWrapper, snake_factory, 100, 128, 0.1, 0.05, save_folder="snake/models", save_rate=0.2)