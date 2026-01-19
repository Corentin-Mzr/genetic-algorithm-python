from typing import Type

import numpy as np
import matplotlib.pyplot as plt

from core.genetic_algorithm import GeneticAlgorithm, NNFactoryFunc, NeuralNetwork
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
    save = max(1, int(1 / save_rate))
    
    best_ai_any_gen: NeuralNetwork | None = None
    best_fitness_any_gen: float = -np.inf
    
    for gen in range(generations):
        best_ai = ga.evolve(parallel=True)
        
        # Re-evaluate
        best_ai.fitness = ga.play(best_ai, env)
        
        if best_ai.fitness > best_fitness_any_gen:
            best_ai_any_gen = best_ai
            best_fitness_any_gen = best_ai.fitness
        
        if gen % save == 0:
            filepath = f"{save_folder}/best_ai_gen_{ga.generation:03d}.npy"
            print(f"Saved {filepath} | Fitness: {best_ai.fitness:.1f}")
            np.save(filepath, best_ai.get_weights())
            
    # Final save
    filepath = f"{save_folder}/best_ai_gen_{ga.generation:03d}.npy"
    print(f"Saved {filepath} | Fitness: {best_ai.fitness:.1f}")
    np.save(filepath, best_ai.get_weights())
    
    # Save best ever
    if best_ai_any_gen is not None:
        filepath = f"{save_folder}/best_ai.npy"
        print(f"Saved {filepath} | Fitness: {best_fitness_any_gen:.1f}")
        np.save(filepath, best_ai_any_gen.get_weights())
            
            
    # Plot result with matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), sharex=True, linewidth=3)
    axs.semilogy(ga.best_fitnesses, label="Best fitness")
    axs.semilogy(ga.avg_fitnesses, label="Average fitness")
    axs.set_title("Fitness evolution")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Fitness")
    axs.grid(True, alpha=0.3)
    axs.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_folder}/training.svg")
