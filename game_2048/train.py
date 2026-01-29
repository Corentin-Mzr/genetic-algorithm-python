from game_2048.wrapper import Game2048Wrapper
from game_2048.ai import Game2048AI

from core.train import train, save_plot, save_training

if __name__ == '__main__':
    def game_2048_factory() -> Game2048AI:
        return Game2048AI(input_size=Game2048Wrapper().input_size, hidden_size=4, output_size=Game2048Wrapper().output_size)
    
    save_folder = "game_2048/models"
    result = train(Game2048Wrapper, game_2048_factory, 200, 128, 0.2, 0.1, save_rate=0.2)
    save_training(result, save_folder)
    save_plot(result, f"{save_folder}/training.svg")