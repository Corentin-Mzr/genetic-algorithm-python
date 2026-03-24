from steering.wrapper import SteeringWrapper
from steering.ai import SteeringAI

from core.train import train, save_plot, save_training, GATrainConfig

if __name__ == '__main__':
    def steering_factory() -> SteeringAI:
        return SteeringAI(input_size=SteeringWrapper().input_size, hidden_size=8, output_size=SteeringWrapper().output_size)
    
    config = GATrainConfig(
        generations=50,
        pop_size=256,
        mutation_rate=0.2,
        mutation_strength=0.1,
        num_trials=1,
        parallel=True,
        elite_ratio=0.06,
        tournament_size=3,
        tournament_ratio=0.5,
        crossover_points=6,
        offspring_ratio=0.9
    )
    
    
    save_folder = "steering/models"
    result = train(SteeringWrapper, steering_factory, save_rate=0.2, config=config)
    save_training(result, save_folder)
    save_plot(result, f"{save_folder}/training.svg")
