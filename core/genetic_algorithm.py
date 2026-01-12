import os
from typing import Type, Callable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial

import numpy as np

from core.environment import Environment
from core.neural_network import NeuralNetwork

NNFactoryFunc = Callable[[], NeuralNetwork]
FitnessFunc = Callable[[Environment, NeuralNetwork], float]

np.random.seed(os.getpid())

class GeneticAlgorithm:
    def __init__(self, 
                 env_type: Type[Environment], 
                 nn_factory: NNFactoryFunc,
                 population_size: int, 
                 mutation_rate: float, 
                 mutation_strength: float):
        self.env = env_type
        self.nn_factory = nn_factory
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation: int = 0
        
        self.population: list[NeuralNetwork] = [self.nn_factory() for _ in range(population_size)]

        self.best_fitnesses: list[float] = []
        self.avg_fitnesses: list[float] = []
        
    @staticmethod
    def play(ai: NeuralNetwork, env_type: Type[Environment]) -> float:
        """ Make the AI do actions in the environment until termination """
        env = env_type()
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_idx = ai.get_action(state)
            state, reward, done = env.step(action_idx)
            total_reward += reward
            
        return total_reward
        
    def evaluate(self) -> None:
        """ Evaluate an AI on multiple samples """
        for ai in self.population:
            scores: list[float] = []
            for _ in range(10):
                score = self.play(ai, self.env)
                scores.append(score)
            ai.fitness = np.maximum(0.0, np.mean(scores).astype(float))
    
    def selection(self) -> list[NeuralNetwork]:
        """ Select best AIs during this generation """
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:self.population_size // 2]
    
    def crossover(self, p1: NeuralNetwork, p2: NeuralNetwork) -> NeuralNetwork:
        """ Crossover to create new generation """
        child = self.nn_factory()
        
        w1 = p1.get_weights()
        w2 = p2.get_weights()
        
        mask = np.random.rand(len(w1)) < 0.5
        child_weights = np.where(mask, w1, w2)
        child.set_weights(child_weights)
        
        return child
    
    def mutate(self, ai: NeuralNetwork) -> None:
        """ Add mutation i.e. randomly change weights """
        weights = ai.get_weights()
        mask = np.random.rand(len(weights)) < self.mutation_rate
        weights[mask] += np.random.randn(np.sum(mask.astype(int))) * self.mutation_strength 
        
        ai.set_weights(weights)
        
    def evolve(self) -> NeuralNetwork:
        """ Create the new generation """
        self.evaluate()
        parents = self.selection()
        new_pop = deepcopy(parents)
        
        while len(new_pop) < self.population_size:
            i1 = np.random.randint(0, len(parents), size=6)
            p1 = parents[np.min(i1)]
            
            i2 = np.random.randint(0, len(parents), size=6)
            p2 = parents[np.min(i2)]
            
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_pop.append(child)
            
        self.population = new_pop
        self.generation += 1
        
        best_fitness = parents[0].fitness
        avg_fitness = np.mean([ai.fitness for ai in parents])
        
        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(avg_fitness.astype(float))
        
        print(f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f}")
        return parents[0]
    
    def evolve_parallel(self) -> NeuralNetwork:
        """ Create the new generation """
        # Evaluate
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            worker = partial(GeneticAlgorithm.play, env_type=self.env)
            fitnesses = list(executor.map(worker, self.population))
            
        for indiv, fitness in zip(self.population, fitnesses):
            indiv.fitness = max(0.0, fitness)
            
        # Selection
        parents = self.selection()
        new_pop = deepcopy(parents)
        
        # Crossover + Mutation
        while len(new_pop) < self.population_size:
            i1 = np.random.randint(0, len(parents), size=6)
            p1 = parents[np.min(i1)]
            
            i2 = np.random.randint(0, len(parents), size=6)
            p2 = parents[np.min(i2)]
            
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_pop.append(child)
            
        self.population = new_pop
        self.generation += 1
        
        best_ai = max(self.population, key=lambda ai: ai.fitness)
        best_fitness = best_ai.fitness
        avg_fitness = np.mean([ai.fitness for ai in parents])
        
        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(avg_fitness.astype(float))
        
        print(f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f}")
        return best_ai     
        
        
if __name__ == '__main__':
    # import sys
    # from pathlib import Path
    
    # project_root = Path(__file__).parent.parent
    # sys.path.insert(0, str(project_root))
    pass
    
    
        