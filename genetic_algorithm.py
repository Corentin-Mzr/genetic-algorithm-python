from snake_ai import SnakeAI
from snake_game import SnakeGame, Direction, GameState, DirectionRelative
from constants import GRID_HEIGHT, GRID_WIDTH, Position

import numpy as np
import os
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

# np.random.seed(42)
np.random.seed(os.getpid())

def generate_random_snake() -> List[Position]:
    # Random snake length
    length = np.random.randint(1, GRID_WIDTH * GRID_HEIGHT // 2)
    
    # Start with a random position for the head
    head_x = np.random.randint(0, GRID_WIDTH)
    head_y = np.random.randint(0, GRID_HEIGHT)
    
    snake = [(head_x, head_y)]
    occupied = {(head_x, head_y)}
    
    # Possible directions: right, left, down, up
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # Grow the snake segment by segment
    for _ in range(length - 1):
        current_x, current_y = snake[-1]
        
        # Shuffle directions to try them in random order
        np.random.shuffle(directions)
        
        # Try each direction until we find a valid one
        placed = False
        for dx, dy in directions:
            new_x = current_x + dx
            new_y = current_y + dy
            
            # Check if the new position is valid
            if (0 <= new_x < GRID_WIDTH and 
                0 <= new_y < GRID_HEIGHT and 
                (new_x, new_y) not in occupied):
                
                snake.append((new_x, new_y))
                occupied.add((new_x, new_y))
                placed = True
                break
        
        # If we couldn't place a segment, return what we have
        if not placed:
            break
    
    return snake

def play_snake(ai: SnakeAI, max_steps: int = 3000) -> Tuple[float, float]:
    game: SnakeGame = SnakeGame()
    game.prev_direction = Direction.UP
    
    # Randomize snake size and position
    # game.snake = generate_random_snake()
    
    # Avoid collision with snake
    # game.spawn_apple()
        
    steps: int = 0
    steps_no_eat: int = 0
    
    while steps < max_steps and steps_no_eat < 300:
        state: GameState = game.get_state()
        direction: DirectionRelative = ai.get_action(state.to_array())
        
        game.move_snake_relative(direction)
        steps += 1
        steps_no_eat += 1
        
        if game.is_apple_eaten():
            game.add_snake_part()
            game.increment_score()
            game.spawn_apple()
            steps_no_eat = 0
            
        if game.is_snake_colliding() or game.is_win():
            break
    
    eat_reward: float = game.score ** 2
    survival_reward: float = steps * 0.25
    length_reward: float = len(game.snake) * 0.1
    
    collision_penalty: float = -400 if game.is_snake_colliding() else 0
    too_long_survival_penalty: float = -0.0 * steps_no_eat
    
    # game.score ** 2 * 200 + steps * 0.75 - 1.5 * steps_no_eat #1.05 ** steps_no_eat
    fitness: float = eat_reward + survival_reward + length_reward + collision_penalty + too_long_survival_penalty 
    return max(0, fitness), game.score

def evaluate_individual(ai: SnakeAI) -> Tuple[float, float]:
    return play_snake(ai)

class GeneticAlgorithm:
    def __init__(self, population_size: int, mutation_rate: float, mutation_strength: float):
        self.population_size: int = population_size
        self.mutation_rate: float = mutation_rate
        self.mutation_strength: float = mutation_strength
        
        self.population: List[SnakeAI] = [SnakeAI() for _ in range(population_size)]
        self.generation: int = 0
        
        self.best_fitnesses: List[float] = []
        self.avg_fitnesses: List[float] = []
        self.best_scores: List[float] = []
        self.avg_scores: List[float] = []
        
    def evaluate(self) -> None:
        for ai in self.population:
            scores: List[float] = []
            apples: List[float] = []
            for _ in range(10):
                score, apple = play_snake(ai)
                scores.append(score)
                apples.append(apple)
            ai.fitness = np.mean(scores).astype(float)
            ai.raw_score = np.mean(apples).astype(float)
    
    def selection(self) -> List[SnakeAI]:
        # Select best AIs during this generation
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:self.population_size // 2]
    
    def crossover(self, p1: SnakeAI, p2: SnakeAI) -> SnakeAI:
        # Crossover to create new generation
        child: SnakeAI = SnakeAI()
        
        w1: np.ndarray = p1.get_weights()
        w2: np.ndarray = p2.get_weights()
        
        mask: np.ndarray = np.random.rand(len(w1)) < 0.5
        child_weights: np.ndarray = np.where(mask, w1, w2)
        child.set_weights(child_weights)
        
        return child
    
    def mutate(self, ai: SnakeAI) -> None:
        # Add mutation i.e. randomly change weights
        weights: np.ndarray = ai.get_weights()
        mask = np.random.rand(len(weights)) < self.mutation_rate
        weights[mask] += np.random.randn(np.sum(mask.astype(int))) * self.mutation_strength 
        
        ai.set_weights(weights)
        
    def evolve(self) -> SnakeAI:
        # Create the new generation
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
        
        best_score = parents[0].raw_score
        avg_score = np.mean([ai.raw_score for ai in parents])
        
        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(avg_fitness.astype(float))
        self.best_scores.append(best_score)
        self.avg_scores.append(avg_score.astype(float))
        
        print(f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f} | Best Score {best_score:.1f} | Avg Score {avg_score:.1f}")
        return parents[0]
    
    def evolve_parallel(self) -> SnakeAI:
        # Evaluate
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            fitnesses = list(executor.map(evaluate_individual, self.population))
            
        for indiv, fitness in zip(self.population, fitnesses):
            indiv.fitness = fitness[0]
            indiv.raw_score = fitness[1]
            
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
        
        best_score = best_ai.raw_score
        avg_score = np.mean([ai.raw_score for ai in parents])
        
        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(avg_fitness.astype(float))
        self.best_scores.append(best_score)
        self.avg_scores.append(avg_score.astype(float))
        
        print(f"Gen {self.generation} | Best Fitness {best_fitness:.1f} | Avg Fitness {avg_fitness:.1f} | Best Score {best_score:.1f} | Avg Score {avg_score:.1f}")
        return best_ai
            
        
        
if __name__ == '__main__':
    pass
        