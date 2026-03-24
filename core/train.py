from typing import Type
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from core.genetic_algorithm import GeneticAlgorithm, NNFactoryFunc, NeuralNetwork
from core.environment import Environment


@dataclass(init=True, frozen=True, slots=True)
class GATrainConfig:
    generations: int = 100
    pop_size: int = 128
    mutation_rate: float = 0.2
    mutation_strength: float = 0.1
    num_trials: int = 40
    parallel: bool = True
    elite_ratio: float = 0.1
    tournament_size: int = 3
    tournament_ratio: float = 0.5
    crossover_points: int = 4
    offspring_ratio: float = 0.9
    early_exit_enable: bool = True
    early_exit_avg_over_n: int = 10
    early_exit_stop_after_n: int = 20
    early_exit_threshold: float = 0.05


@dataclass(frozen=True, slots=True)
class GATrainResult:
    best_ai: NeuralNetwork
    best_fitness: float
    best_ai_per_gen: list[NeuralNetwork]
    saved_gen: list[int]
    generations: int
    best_fitness_per_gen: list[float]
    average_fitness_per_gen: list[float]


def save_plot(
    result: GATrainResult, filepath: str = "training.svg", plot_style: str = "default"
) -> None:
    """
    Save the plot that shows fitness evolution during training

    Plot style can be default or log
    """
    styles = ["default", "log"]
    if plot_style not in styles:
        raise ValueError(f"Invalid plot style. Must be one of {styles}")

    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), sharex=True, linewidth=3)

    if plot_style == "log":
        axs.semilogy(result.best_fitness_per_gen, label="Best fitness")
        axs.semilogy(result.average_fitness_per_gen, label="Average fitness")
    else:
        axs.plot(result.best_fitness_per_gen, label="Best fitness")
        axs.plot(result.average_fitness_per_gen, label="Average fitness")

    axs.set_title("Fitness evolution")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Fitness")
    axs.grid(True, alpha=0.3)
    axs.legend()

    plt.tight_layout()
    plt.savefig(filepath)

    print("Fitness plot saved")


def save_training(result: GATrainResult, save_folder: str = ".") -> None:
    """Save trained neural networks"""
    for i in range(len(result.best_ai_per_gen)):
        gen = result.saved_gen[i]
        fitness = result.best_fitness_per_gen[gen]

        filepath = f"{save_folder}/best_ai_gen_{result.saved_gen[i]:03d}.npy"
        np.save(filepath, result.best_ai_per_gen[i].get_weights())
        print(f"Saved {filepath} | Fitness: {fitness:.1f}")

    filepath = f"{save_folder}/best_ai.npy"
    np.save(filepath, result.best_ai.get_weights())
    print(f"Saved {filepath} | Fitness: {result.best_fitness:.1f}")


def train(
    env: Type[Environment],
    nn_factory: NNFactoryFunc,
    save_rate: float = 0.1,
    config: GATrainConfig | None = None,
) -> GATrainResult:
    """Train a neural network using a genetic algorithm"""

    if config is None:
        config = GATrainConfig()
    assert config.generations > 0

    with GeneticAlgorithm(
        env, nn_factory, config.pop_size, config.mutation_rate, config.mutation_strength
    ) as ga:

        save = max(1, int(1 / save_rate))

        best_ai_global = nn_factory()
        best_fitness_global = -np.inf

        best_ai_per_gen: list[NeuralNetwork] = []
        saved_gen: list[int] = []

        curr_best_fitness = -np.inf
        early_count = 0
        exited_early = False

        for gen in range(config.generations):
            best_ai_this_gen = ga.evolve(
                config.num_trials,
                config.parallel,
                config.elite_ratio,
                config.tournament_size,
                config.tournament_ratio,
                config.offspring_ratio,
                config.crossover_points,
            )

            curr_best_fitness = best_ai_this_gen.fitness

            best_ai_this_gen.fitness = ga.best_fitnesses[-1]

            if curr_best_fitness > best_fitness_global:
                best_ai_global = best_ai_this_gen
                best_fitness_global = curr_best_fitness
                print("New best AI found")

            if gen % save == 0:
                saved_gen.append(gen)
                best_ai_per_gen.append(best_ai_this_gen)

            if not config.early_exit_enable:
                continue

            if gen >= config.early_exit_avg_over_n * 2:
                prev_avg = np.mean(
                    ga.best_fitnesses[
                        -2
                        * config.early_exit_avg_over_n : -config.early_exit_avg_over_n
                    ]
                )
                curr_avg = np.mean(ga.best_fitnesses[-config.early_exit_avg_over_n :])

                if np.abs(curr_avg - prev_avg) < config.early_exit_threshold * abs(
                    prev_avg
                ):
                    early_count += 1
                    print(
                        f"Improvement not significant {early_count} | Previous Average {prev_avg:.0f} | Current Average {curr_avg:.0f}"
                    )
                elif curr_best_fitness < best_fitness_global:
                    early_count += 1
                    print(
                        f"Worse performance {early_count} | Best Fitness Global {best_fitness_global:.0f} | Current Best Fitness {curr_best_fitness:.0f}"
                    )
                else:
                    early_count = 0

            if early_count >= config.early_exit_stop_after_n:
                exited_early = True
                break

        if exited_early:
            print("Early exit")

        result = GATrainResult(
            best_ai=best_ai_global,
            best_fitness=best_fitness_global,
            best_ai_per_gen=best_ai_per_gen,
            saved_gen=saved_gen,
            generations=gen if exited_early else config.generations,
            best_fitness_per_gen=ga.best_fitnesses,
            average_fitness_per_gen=ga.avg_fitnesses,
        )

        return result
