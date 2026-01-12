from typing import Type

import numpy as np
import matplotlib.pyplot as plt

from core.genetic_algorithm import GeneticAlgorithm, NNFactoryFunc
from core.environment import Environment

def train(env: Type[Environment], 
          nn_factory: NNFactoryFunc, 
          generations: int,
          pop_size: int, 
          mutation_rate: float, 
          mutation_strength: float,
          save_folder: str = ".",
          save_rate: float = 0.1
    ) -> None:
    
    ga = GeneticAlgorithm(env, nn_factory, pop_size, mutation_rate, mutation_strength)
    save = int(generations * save_rate)
    
    for gen in range(generations):
        best_ai = ga.evolve_parallel()
        
        if gen % save == 0:
            filepath = f"{save_folder}/best_ai_gen_{gen:03d}.npy"
            print(f"Saved {filepath}")
            np.save(filepath, best_ai.get_weights())
            
    # Final save
    filepath = f"{save_folder}/best_ai_gen_{gen:03d}.npy"
    print(f"Saved {filepath}")
    np.save(filepath, best_ai.get_weights())
            
            
    # Plot result with matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), sharex=True, linewidth=3)
    axs.plot(ga.best_fitnesses, label="Best fitness")
    axs.plot(ga.avg_fitnesses, label="Average fitness")
    axs.set_title("Fitness evolution")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Fitness")
    axs.grid(True, alpha=0.3)
    axs.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_folder}/training.svg")
