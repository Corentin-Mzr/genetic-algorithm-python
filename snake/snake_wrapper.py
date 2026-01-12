import numpy as np

from core.environment import Environment, Observation
from snake_game import SnakeGame, RELATIVE_ACTIONS, GRID_WIDTH, GRID_HEIGHT, Position


class SnakeGameWrapper(Environment):
    def __init__(self):
        self.game = SnakeGame()
        self.steps_without_eating = 0
        self.total_steps = 0
        
    @property
    def input_size(self) -> int:
        return 24
    
    @property
    def output_size(self) -> int:
        return 3
    
    @staticmethod
    def _generate_random_snake() -> list[Position]:
        """ Generate a snake of random size in a random position """
        length = np.random.randint(1, GRID_WIDTH * GRID_HEIGHT // 2)
        head_x = np.random.randint(0, GRID_WIDTH)
        head_y = np.random.randint(0, GRID_HEIGHT)
        
        snake = [(head_x, head_y)]
        occupied = {(head_x, head_y)}
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for _ in range(length - 1):
            current_x, current_y = snake[-1]
            
            np.random.shuffle(directions)
            
            placed = False
            for dx, dy in directions:
                new_x = current_x + dx
                new_y = current_y + dy
                
                if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in occupied):
                    snake.append((new_x, new_y))
                    occupied.add((new_x, new_y))
                    placed = True
                    break

            if not placed:
                break
        
        return snake
    
    def reset(self) -> np.ndarray:
        self.game = SnakeGame()
        # self.game.snake = self._generate_random_snake()
        return self.game.get_state().to_array()
    
    def step(self, action_idx: int) -> Observation:
        # Compute prev distance to apple
        hy, hx = self.game.snake[0]
        ay, ax = self.game.apple
        prev_dist = abs(hy - ay) + abs(hx - ax)
        
        # Move
        action = RELATIVE_ACTIONS[action_idx]
        self.game.move_relative(action)
        
        # Compute new distance to apple
        hy, hx = self.game.snake[0]
        new_dist = abs(hy - ay) + abs(hx - ax)
        
        reward = 0
        done = False
        self.steps_without_eating += 1
        self.total_steps += 1
        
        # Must be efficient
        reward -= 0.05
        
        # Getting close to food is good
        if new_dist < prev_dist:
            reward += 1.0
        else:
            reward -= 1.5 * self.steps_without_eating
        
        # Eating is good
        if self.game.just_ate:
            self.steps_without_eating = 0
            reward = 100
        
        # Starvation is bad
        if self.steps_without_eating > 100 + 10 * len(self.game.snake):
            done = True
            reward = -100

        # Collision is bad            
        if self.game.is_colliding():
            done = True
            reward = -150
            
        # Surviving is good
        # if self.total_steps > 5000:
        #     done = True
        #     reward = 50
        
        # Winning is very good    
        if self.game.is_win():
            done = True
            reward = 10000
            
        return self.game.get_state().to_array(), reward, done
    
    
if __name__ == '__main__':
    wrapper = SnakeGameWrapper()
        