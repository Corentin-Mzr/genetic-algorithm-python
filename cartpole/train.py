from cartpole.wrapper import CartpoleWrapper
from cartpole.ai import CartpoleAI

from core.train import train, save_plot, save_training, GATrainConfig

if __name__ == '__main__':
    def cartpole_factory() -> CartpoleAI:
        return CartpoleAI(input_size=CartpoleWrapper().input_size, hidden_size=2, output_size=CartpoleWrapper().output_size)
    
    config = GATrainConfig(
        generations=50,
        pop_size=128,
        mutation_rate=0.2,
        mutation_strength=0.1,
        num_trials=20,
        parallel=True,
        elite_ratio=0.06,
        tournament_size=3,
        tournament_ratio=0.5,
        crossover_points=6,
        offspring_ratio=0.9
    )
    
    
    save_folder = "cartpole/models"
    result = train(CartpoleWrapper, cartpole_factory, save_rate=0.2, config=config)
    save_training(result, save_folder)
    save_plot(result, f"{save_folder}/training.svg")
