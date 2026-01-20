from typing import Type
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from core.genetic_algorithm import GeneticAlgorithm, NNFactoryFunc, NeuralNetwork
from core.environment import Environment

@dataclass
class TrainResult:
    best_ai: NeuralNetwork
    best_fitness: float
    
    best_ai_per_gen: list[NeuralNetwork]
    saved_gen: list[int]
    generations: int
    
    best_fitness_per_gen: list[float]
    average_fitness_per_gen: list[float]
    
def save_plot(result: TrainResult, filepath: str = "training.svg") -> None:
    """ Save the plot that shows fitness evolution during training """
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), sharex=True, linewidth=3)
    axs.semilogy(result.best_fitness_per_gen, label="Best fitness")
    axs.semilogy(result.average_fitness_per_gen, label="Average fitness")
    axs.set_title("Fitness evolution")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Fitness")
    axs.grid(True, alpha=0.3)
    axs.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    
    print("Fitness plot saved")
    
def save_training(result: TrainResult, save_folder: str = ".") -> None:
    """ Save trained neural networks """
    for i in range(len(result.best_ai_per_gen)):
        filepath = f"{save_folder}/best_ai_gen_{result.saved_gen[i]:03d}.npy"
        np.save(filepath, result.best_ai_per_gen[i].get_weights())
        print(f"Saved {filepath} | Fitness: {result.best_fitness_per_gen[i]:.1f}")
        
    filepath = f"{save_folder}/best_ai.npy"
    np.save(filepath, result.best_ai.get_weights())
    print(f"Saved {filepath} | Fitness: {result.best_fitness:.1f}")
    

def train(env: Type[Environment], 
          nn_factory: NNFactoryFunc, 
          generations: int,
          pop_size: int, 
          mutation_rate: float, 
          mutation_strength: float,
          save_rate: float = 0.1
    ) -> TrainResult:
    """ Train a neural network using a genetic algorithm """
    
    ga = GeneticAlgorithm(env, nn_factory, pop_size, mutation_rate, mutation_strength)
    save = max(1, int(1 / save_rate))
    
    best_ai: NeuralNetwork = nn_factory()
    best_fitness: float = -np.inf
    
    best_ai_per_gen: list[NeuralNetwork] = []
    saved_gen: list[int] = []
    
    for gen in range(generations):
        best_ai = ga.evolve(parallel=True)
        
        # Re-evaluate
        best_ai.fitness = ga.play(best_ai, env, num_trials=40)
        
        if best_ai.fitness > best_fitness:
            best_ai = best_ai
            best_fitness = best_ai.fitness
            
        if gen % save == 0:
            saved_gen.append(gen)
            best_ai_per_gen.append(best_ai)
            
    result = TrainResult(
        best_ai=best_ai,
        best_fitness=best_fitness,
        best_ai_per_gen=best_ai_per_gen,
        saved_gen=saved_gen,
        generations=generations,
        best_fitness_per_gen=ga.best_fitnesses,
        average_fitness_per_gen=ga.avg_fitnesses,
    )
    
    return result
