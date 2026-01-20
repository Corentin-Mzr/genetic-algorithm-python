import numpy as np

from core.environment import Environment, Observation
from snake_game import SnakeGame, Position, Direction
from snake_game import GRID_WIDTH, GRID_HEIGHT, RELATIVE_ACTIONS, ABSOLUTE_TO_RELATIVE

from copy import deepcopy


class SnakeGameWrapper(Environment):
    def __init__(self):
        self.game = SnakeGame()
        self.danger = self._count_danger_moves(self.game.direction, self.game.snake, self.game.just_ate)
        
    @property
    def input_size(self) -> int:
        return 28
    
    @property
    def output_size(self) -> int:
        return 3
    
    def _generate_random_snake(self) -> list[Position]:
        """ Generate a snake of random size in a random position """
        length = np.random.randint(1, GRID_WIDTH * GRID_HEIGHT // 2)
        head_x = np.random.randint(0, GRID_WIDTH)
        head_y = np.random.randint(0, GRID_HEIGHT)
        
        snake = [(head_y, head_x)]
        occupied = {(head_y, head_x)}
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for _ in range(length - 1):
            current_y, current_x = snake[-1]
            
            np.random.shuffle(directions)
            
            placed = False
            for dx, dy in directions:
                new_x = current_x + dx
                new_y = current_y + dy
                
                if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_y, new_x) not in occupied):
                    snake.append((new_y, new_x))
                    occupied.add((new_y, new_x))
                    placed = True
                    break

            if not placed:
                break
        
        return snake
    
    def _will_colide(self, pos: Position, snake: list[Position], just_ate: bool) -> bool:
        """ Check if there is a collision at the given position"""
        y, x = pos
        if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
            return True
        return pos in snake[:-1] if not just_ate else pos in snake
    
    def _count_danger_moves(self, direction: Direction, snake: list[Position], just_ate: bool) -> int:
        """ Based on given position and direction, count the number of moves that would result in a collision next turn """
        danger = 0
        head = snake[0]
        for action in RELATIVE_ACTIONS:
            dy, dx = ABSOLUTE_TO_RELATIVE[direction][action]
            ny, nx = head[0] + dy, head[1] + dx
            if self._will_colide((ny, nx), snake, just_ate):
                danger += 1
        return danger
    
    def _simulate_move(self, direction: Direction) -> tuple[list[Position], bool]:
        """ Simulate a move in the given direction and return the new snake positions """
        y, x = self.game.snake[0]
        snake = deepcopy(self.game.snake)
        just_ate = False
        
        match direction:
            case Direction.UP:
                new_head = (y - 1, x)
            case Direction.LEFT: 
                new_head = (y, x - 1)
            case Direction.RIGHT:
                new_head = (y, x + 1)
            case Direction.DOWN:
                new_head = (y + 1, x)
                
        snake.insert(0, new_head)
        
        if new_head == self.game.apple:
            just_ate = True
        else:
            snake.pop()
            
        return snake, just_ate
    
    def reset(self) -> np.ndarray:
        self.game = SnakeGame()
        self.danger = self._count_danger_moves(self.game.direction, self.game.snake, self.game.just_ate)
        return self.game.get_state().to_array()
    
    def step(self, action_idx: int) -> Observation:
        reward = 0
        done = False
        action = RELATIVE_ACTIONS[action_idx]
        
        # Reward and penalty types
        avoidance_reward = 0
        eating_reward = 0
        efficiency_penalty = 0
        starvation_penalty = 0
        collision_penalty = 0
        winning_reward = 0
        curriculum_reward = 0
        
        # Avoidance reward
        # current_danger = self._count_danger_moves(self.game.direction, self.game.snake, self.game.just_ate)
        # if current_danger < self.danger:
        #     danger_level = self.danger / len(RELATIVE_ACTIONS)
        #     avoidance_reward = 20 + 2 * danger_level * len(self.game.snake) ** 1.2
        # self.danger = current_danger
        
        # Move
        self.game.move_relative(action)
        
        # Reward if danger is reduced, reward more if snake is long
        # new_danger = self._count_danger_moves(self.game.direction, self.game.snake, self.game.just_ate)
        # if new_danger < current_danger:
        #     danger_level = current_danger / len(RELATIVE_ACTIONS)
        #     avoidance_reward = 20 + 2 * danger_level * len(self.game.snake) ** 1.2
        # avoidance_reward = -20 * new_danger
        
        # Must be efficient
        # efficiency_penalty = -0.1 * self.game.steps_without_eating # -1.0
        
        # Eating is good
        if self.game.just_ate:
            eating_reward = (50 + 10 * self.game.score ** 2) * (1 / (1 + self.game.steps_without_eating ** 0.5))
        
        # Starvation is bad
        if self.game.steps_without_eating > 50 + 2 * len(self.game.snake):
            done = True
            starvation_penalty = -50

        # Collision is bad            
        if self.game.is_colliding():
            done = True
            collision_penalty = -100
        
        # Winning is very good    
        if self.game.is_win():
            done = True
            winning_reward = 10000
         
        # Curriculum learning    
        if done:
            curriculum_reward = self.game.score * 20
            
        # Total reward
        reward = avoidance_reward + eating_reward + efficiency_penalty + starvation_penalty + collision_penalty + winning_reward + curriculum_reward
            
        return self.game.get_state().to_array(), reward, done
    
    
if __name__ == '__main__':
    wrapper = SnakeGameWrapper()
    state = wrapper.reset()
    print(state)
    