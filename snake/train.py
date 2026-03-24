from snake.ai import SnakeAI
from snake.wrapper import SnakeGameWrapper
from core.train import train, save_training, save_plot, GATrainConfig

if __name__ == '__main__':
    def snake_factory() -> SnakeAI:
        return SnakeAI(input_size=SnakeGameWrapper().input_size, hidden_size=7, output_size=SnakeGameWrapper().output_size)
    
    config = GATrainConfig(
        generations=500,
        pop_size=128,
        mutation_rate=0.2,
        mutation_strength=0.1,
        num_trials=10,
        parallel=True,
        elite_ratio=0.06,
        tournament_size=3,
        tournament_ratio=0.5,
        crossover_points=6,
        offspring_ratio=0.9
    )
    
    save_folder = "snake/models"
    result = train(SnakeGameWrapper, snake_factory, save_rate=0.2, config=config)
    save_training(result, save_folder)
    save_plot(result, f"{save_folder}/training.svg")
    