import numpy as np

from core.environment import Environment, Observation
from snake_game import SnakeGame, Position, Direction
from snake_game import GRID_WIDTH, GRID_HEIGHT, RELATIVE_ACTIONS, ABSOLUTE_TO_RELATIVE, LEFT_DIRECTIONS, RIGHT_DIRECTIONS, DirectionRelative

from copy import deepcopy


class SnakeGameWrapper(Environment):
    def __init__(self):
        self.game = SnakeGame()
        self.generation = 0
        
    @property
    def input_size(self) -> int:
        return 28
    
    @property
    def output_size(self) -> int:
        return 3
    
    def _generate_random_snake(self) -> list[Position]:
        """ Generate a snake of random size in a random position """
        length = np.random.randint(1, min(self.generation // 4 + 2, GRID_WIDTH * GRID_HEIGHT // 2))
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
    
    # def _predict_position(self, action: Direction) -> Position:
    #     """ Predict the new head position after taking the given action"""
    #     y, x = self.game.snake[0]
    #     dy, dx = DIRECTION_TO_POSITION[action]
    #     return (y + dy, x + dx)
    
    # def _count_danger_moves(self, head: Position) -> int:
    #     """ Count the number of moves that would result in a collision, starting from the given head position"""
    #     danger = 0
    #     for action in ABSOLUTE_ACTIONS:
    #         dy, dx = DIRECTION_TO_POSITION[action]
    #         ny, nx = head[0] + dy, head[1] + dx
    #         if self._will_colide((ny, nx)):
    #             danger += 1
    #     return danger
    
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
        
        # # Avoidance reward
        # current_danger = self._count_danger_moves(self.game.direction, self.game.snake, self.game.just_ate)
        # if action == DirectionRelative.LEFT:
        #     new_direction = LEFT_DIRECTIONS[self.game.direction]
        # elif action == DirectionRelative.RIGHT:
        #     new_direction = RIGHT_DIRECTIONS[self.game.direction]
        # else:
        #     new_direction = self.game.direction
        # new_snake, new_ate = self._simulate_move(new_direction)
        # new_danger = self._count_danger_moves(new_direction, new_snake, new_ate)
        
        # # Reward if danger is reduced, reward more if snake is long
        # if new_danger < current_danger:
        #     danger_level = current_danger / len(RELATIVE_ACTIONS)
        #     avoidance_reward = 10 * danger_level + 2 * len(self.game.snake) ** 1.2
        
        # Move
        self.game.move_relative(action)
        
        # Must be efficient
        # efficiency_penalty = -1.0
        
        # Eating is good
        if self.game.just_ate:
            eating_reward = (50 + 10 * self.game.score ** 2) * (1 / (1 + self.game.steps_without_eating))
        
        # Starvation is bad
        if self.game.steps_without_eating > 50 + 5 * len(self.game.snake):
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
            curriculum_reward = self.game.score * 100 # - 1.25 * self.steps_without_eating
            self.generation += 1
            
        # reward_dict = {
        #     "avoidance_reward": avoidance_reward,
        #     "eating_reward": eating_reward,
        #     "efficiency_penalty": efficiency_penalty,
        #     "starvation_penalty": starvation_penalty,
        #     "collision_penalty": collision_penalty,
        #     "winning_reward": winning_reward,
        #     "curriculum_reward": curriculum_reward
        # }
        # print(reward_dict)
            
        # Total reward
        reward = avoidance_reward + eating_reward + efficiency_penalty + starvation_penalty + collision_penalty + winning_reward + curriculum_reward
            
        return self.game.get_state().to_array(), reward, done
    
    
if __name__ == '__main__':
    wrapper = SnakeGameWrapper()
        
        
    # self.game.snake = self._generate_random_snake()
        
        
    # def _predict_position(self, action: DirectionRelative) -> Position:
    #     y, x = self.game.snake[0]
    #     dy, dx = DIRECTION_TO_RELATIVE[self.game.direction][action]
    #     return (y + dy, x + dx)
    
    # def _next_direction(self, current: Direction, action: DirectionRelative) -> Direction:
    #     if action == DirectionRelative.STRAIGHT:
    #         return LEFT_DIRECTIONS[current]
    #     if action == DirectionRelative.LEFT:
    #         return RIGHT_DIRECTIONS[current]
    #     return current
    
    # def _count_danger_moves(self, head: Position, direction: Direction) -> int:
    #     danger = 0
    #     for action in RELATIVE_ACTIONS:
    #         dy, dx = DIRECTION_TO_RELATIVE[direction][action]
    #         ny, nx = head[0] + dy, head[1] + dx
    #         if self._will_colide((ny, nx)):
    #             danger += 1
    #     return danger
    
    
        # current_direction = self.game.direction
        # current_danger = self._count_danger_moves(current_head, current_direction)
        # danger_level = current_danger / len(RELATIVE_ACTIONS)
        
        # Simulate action
        # new_direction = self._next_direction(current_direction, action)
        # dy, dx = DIRECTION_TO_RELATIVE[current_direction][action]
    
    
    
    # action = RELATIVE_ACTIONS[action_idx]
    
    
        # Check current danger
        # current_danger_actions = 0
        # for action in RELATIVE_ACTIONS:
        #     ny, nx = self._predict_position(action)
        #     if self._will_colide((ny, nx)):
        #         current_danger_actions += 1
                
        # in_danger = current_danger_actions >= 2
        # danger_level = current_danger_actions / len(RELATIVE_ACTIONS)
        
        # # Compare to future danger
        # new_danger_actions = 0
        # new_direction = RELATIVE_ACTIONS[action_idx]
        # ny, nx = self._predict_position(new_direction)
        # for action in RELATIVE_ACTIONS:
        #     dy, dx = DIRECTION_TO_RELATIVE[self.game.direction][action]
        #     nny, nnx = ny + dy, nx + dx
        #     if self._will_colide((nny, nnx)):
        #         new_danger_actions += 1
        
        # danger_avoided = not self._will_colide((ny, nx))
        # danger_reduced = new_danger_actions < current_danger_actions
        
        # # When snake is long, avoiding danger should be rewarded more
        # if in_danger and danger_avoided and danger_reduced:
        #     avoidance_reward = 10 * danger_level + 2 * len(self.game.snake) ** 1.2
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Current state
        # current_head = self.game.snake[0]
        # current_danger = self._count_danger_moves(current_head)
        # danger_level = current_danger / len(ABSOLUTE_ACTIONS)
        # dy, dx = DIRECTION_TO_POSITION[action]
        # new_head = (current_head[0] + dy, current_head[1] + dx)
        
        # if not self._will_colide(new_head):
        #     new_danger = self._count_danger_moves(new_head)
            
        #     # Mobility increased
        #     if new_danger < current_danger:
        #         avoidance_reward = 5 + 2 * danger_level * (current_danger - new_danger)
            
        #     # Escape tight spaces
        #     if current_danger >= 2 and new_danger <= 1:
        #         avoidance_reward += 5
                
        #     # Length scaling
        #     avoidance_reward *= (1 + 0.5 * len(self.game.snake) ** 1.1)