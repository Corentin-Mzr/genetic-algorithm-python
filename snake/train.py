from snake.ai import SnakeAI
from snake.wrapper import SnakeGameWrapper
from core.train import train, save_training, save_plot

if __name__ == '__main__':
    def snake_factory() -> SnakeAI:
        return SnakeAI(input_size=SnakeGameWrapper().input_size, hidden_size=7, output_size=SnakeGameWrapper().output_size)
    
    save_folder = "snake/models"
    result = train(SnakeGameWrapper, snake_factory, 200, 128, 0.2, 0.1, save_rate=0.2)
    save_training(result, save_folder)
    save_plot(result, f"{save_folder}/training.svg")
    