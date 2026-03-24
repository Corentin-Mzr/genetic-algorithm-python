import os
import random
from typing import Type, Self, Callable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from dataclasses import dataclass

import numpy as np

from core.environment import Environment
from core.neural_network import NeuralNetwork

NNFactoryFunc = Callable[[], NeuralNetwork]


class GeneticAlgorithm:
    __slots__ = (
        "env",
        "nn_factory",
        "population_size",
        "mutation_rate",
        "mutation_strength",
        "generation",
        "population",
        "best_fitnesses",
        "avg_fitnesses",
        "executor",
    )

    def __init__(
        self,
        env_type: Type[Environment],
        nn_factory: NNFactoryFunc,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
    ):
        self.env = env_type
        self.nn_factory = nn_factory
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.generation: int = 0

        self.population: list[NeuralNetwork] = [
            self.nn_factory() for _ in range(population_size)
        ]

        self.best_fitnesses: list[float] = []
        self.avg_fitnesses: list[float] = []

        self.executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    @staticmethod
    def play(
        ai: NeuralNetwork,
        env_type: Type[Environment],
        num_trials: int,
        seed: int | None = None,
    ) -> float:
        """Make the AI do actions in the environment until termination"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        rewards: list[float] = []

        for _ in range(num_trials):
            env = env_type()
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action_idx = ai.get_action(state)
                state, reward, done = env.step(action_idx)
                total_reward += reward

            rewards.append(total_reward)
        return float(0.7 * np.median(rewards) + 0.3 * np.max(rewards))

    def evaluate(self, num_trials: int, seed: int | None = None) -> None:
        """Evaluate an AI on multiple samples"""
        for ai in self.population:
            ai.fitness = self.play(ai, self.env, num_trials, seed)

    def selection(
        self, elite_ratio: float, tournament_size: int, tournament_ratio: float
    ) -> list[NeuralNetwork]:
        """Select best AIs during this generation"""
        parents: list[NeuralNetwork] = []

        # Elitism
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        n_elites = max(1, int(self.population_size * elite_ratio))
        elites = self.population[:n_elites]
        parents.extend(elites)

        # Tournament selection
        while len(parents) < int(tournament_ratio * self.population_size):
            candidates = random.sample(self.population, k=tournament_size)
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)

        return [deepcopy(p) for p in parents]  # parents

    def crossover(
        self, p1: NeuralNetwork, p2: NeuralNetwork, n_points: int
    ) -> NeuralNetwork:
        """Crossover to create new generation"""
        child = self.nn_factory()

        w1 = p1.get_weights()
        w2 = p2.get_weights()

        # Multi-point crossover
        n_points = min(n_points, len(w1) - 1)
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
        """Add mutation i.e. randomly change weights"""
        weights = ai.get_weights()

        # Gaussian mutation
        mask = np.random.rand(weights.size) < self.mutation_rate
        weights[mask] += np.random.randn(mask.sum()) * self.mutation_strength

        ai.set_weights(weights)

    def evolve(
        self,
        num_trials: int,
        parallel: bool,
        elite_ratio: float,
        tournament_size: int,
        tournament_ratio: float,
        offspring_ratio: float,
        crossover_points: int,
    ) -> NeuralNetwork:
        """Create the new generation"""
        # Seed
        gen_seed = self.generation

        # Evaluate
        if parallel:

            if self.executor is None:
                max_workers = os.cpu_count() or 1
                self.executor = ProcessPoolExecutor(max_workers=max_workers)

            worker = partial(
                GeneticAlgorithm.play,
                env_type=self.env,
                num_trials=num_trials,
                seed=gen_seed,
            )
            fitnesses = list(self.executor.map(worker, self.population))

            print("Fitnesses ", [round(e, 2) for e in sorted(fitnesses, reverse=True)])

            for indiv, fitness in zip(self.population, fitnesses):
                indiv.fitness = fitness
        else:
            self.evaluate(num_trials, seed=gen_seed)
            fitnesses = [ai.fitness for ai in self.population]

        # Compute avg, retrieve best individual
        best_parent = max(self.population, key=lambda x: x.fitness)
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)

        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(float(avg_fitness))

        print(
            f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f}"
        )

        # Selection
        parents = self.selection(elite_ratio, tournament_size, tournament_ratio)
        new_pop = deepcopy(parents)

        # Crossover + Mutation
        while len(new_pop) < int(offspring_ratio * self.population_size):
            p1 = max(random.sample(parents, k=tournament_size), key=lambda x: x.fitness)
            p2 = p1
            while p2 == p1:
                p2 = max(
                    random.sample(parents, k=tournament_size), key=lambda x: x.fitness
                )

            child = self.crossover(p1, p2, crossover_points)
            self.mutate(child)
            new_pop.append(child)

        # Introduce random new individuals
        while len(new_pop) < self.population_size:
            new_ai = self.nn_factory()
            new_ai.fitness = -np.inf
            new_pop.append(new_ai)

        self.population[:] = new_pop
        self.generation += 1

        return best_parent
