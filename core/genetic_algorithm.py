import os
import random
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
    def play(ai: NeuralNetwork, env_type: Type[Environment], num_trials: int) -> float:
        """ Make the AI do actions in the environment until termination """
        rewards: list[float] = []
        
        for _ in range(num_trials):
            env = env_type()
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action_idx = ai.get_action(state)
                state, reward, done = env.step(action_idx)
                total_reward += reward
            
            rewards.append(total_reward)
            
        return float(0.7 * np.median(rewards) + 0.3 * np.max(rewards))
        
    def evaluate(self, num_trials: int) -> None:
        """ Evaluate an AI on multiple samples """
        for ai in self.population:
            ai.fitness = self.play(ai, self.env, num_trials)
    
    def selection(self) -> list[NeuralNetwork]:
        """ Select best AIs during this generation """
        parents = []
        
        # Elitism
        elite_ratio = 0.02
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        # print([int(pop.fitness) for pop in self.population])
        n_elites = max(1, int(self.population_size * elite_ratio))
        elites = self.population[:n_elites]
        parents.extend(elites)
        
        # Tournament selection
        tournament_size = 2
        tournament_ratio = 0.5
        while len(parents) < int(tournament_ratio * self.population_size):
            candidates = random.sample(self.population, k=tournament_size) #np.random.choice(self.population, size=tournament_size, replace=False)
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)
        
        return [deepcopy(p) for p in parents] # parents
    
    def crossover(self, p1: NeuralNetwork, p2: NeuralNetwork) -> NeuralNetwork:
        """ Crossover to create new generation """
        child = self.nn_factory()
        
        w1 = p1.get_weights()
        w2 = p2.get_weights()
        
        # Multi-point crossover
        n_points = 3
        points = sorted(np.random.choice(len(w1), n_points, replace=False))
        child_weights = w1.copy()
        use_p2 = False
        prev_point = 0
        
        for point in points:
            if use_p2:
                child_weights[prev_point:point] = w2[prev_point:point]
            use_p2 = not use_p2
            prev_point = point
        if use_p2:
            child_weights[prev_point:] = w2[prev_point:]
        
        child.set_weights(child_weights)
        return child
    
    def mutate(self, ai: NeuralNetwork) -> None:
        """ Add mutation i.e. randomly change weights """
        weights = ai.get_weights()
        
        # Gaussian mutation
        mask = np.random.rand(weights.size) < self.mutation_rate
        weights[mask] += np.random.randn(mask.sum()) * self.mutation_strength
        
        ai.set_weights(weights)
        
    def evolve(self, num_trials: int = 40, parallel: bool = False) -> NeuralNetwork:
        """ Create the new generation """
        # Evaluate
        if parallel:
            max_workers = os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                worker = partial(GeneticAlgorithm.play, env_type=self.env, num_trials=num_trials)
                fitnesses = list(executor.map(worker, self.population))
                
            for indiv, fitness in zip(self.population, fitnesses):
                indiv.fitness = fitness
        else:
            self.evaluate(num_trials)
        
        # Selection
        parents = self.selection()
        new_pop = parents.copy()
        
        # Crossover + Mutation
        offspring_ratio = 0.85
        tournament_size = 3
        while len(new_pop) < int(offspring_ratio * self.population_size):
            p1 = max(random.sample(parents, k=tournament_size), key=lambda x: x.fitness)
            p2 = p1
            while p2 == p1:
                p2 = max(random.sample(parents, k=tournament_size), key=lambda x: x.fitness)
            
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_pop.append(child)
            
        # Introduce random new individuals
        while len(new_pop) < self.population_size:
            new_ai = self.nn_factory()
            new_pop.append(new_ai)
            
        self.population = new_pop
        self.generation += 1
        
        best_parent = max(parents, key=lambda x: x.fitness)
        best_fitness = best_parent.fitness
        avg_fitness = np.mean([ai.fitness for ai in parents])
        
        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(float(avg_fitness))
        
        print(f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f}")
        
        # Reset fitness
        for ai in new_pop:
            ai.fitness = 0 # None
        
        return best_parent
        