import numpy as np
import matplotlib.pyplot as plt

from constants import *
from genetic_algorithm import GeneticAlgorithm


def train() -> None:
    ga = GeneticAlgorithm(population_size=128, mutation_rate=0.15, mutation_strength=0.075) # 0.1 0.05
    
    for gen in range(101):
        best_ai = ga.evolve_parallel()
        
        if gen % 10 == 0:
            np.save(f"models/best_ai_gen_{gen:03d}.npy", best_ai.get_weights())
            
            
    # Plot result with matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), sharex=True)
    axs[0].semilogy(ga.best_fitnesses, label="Best fitness")
    axs[0].semilogy(ga.avg_fitnesses, label="Average fitness")
    axs[0].set_title("Fitness evolution")
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Fitness")
    axs[0].legend()
    
    axs[1].plot(ga.best_scores, label="Best score")
    axs[1].plot(ga.avg_scores, label="Average score")
    axs[1].set_title("Score evolution")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Fitness")
    axs[1].legend()
    plt.savefig("models/training.svg")
            


if __name__ == '__main__':
    train()
    
