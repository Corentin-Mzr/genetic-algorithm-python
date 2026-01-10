from typing import List
from enum import Enum
from constants import GRID_WIDTH, GRID_HEIGHT, Position
from dataclasses import dataclass

import numpy as np

class GameObject(Enum):
    AIR = 0
    SNAKE = 1
    APPLE = 2

class Direction(Enum):
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4
    
class DirectionRelative(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    
OPPOSITE_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT
}

LEFT_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT,
    Direction.RIGHT: Direction.UP
}

RIGHT_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.RIGHT,
    Direction.LEFT: Direction.UP,
    Direction.DOWN: Direction.LEFT,
    Direction.RIGHT: Direction.DOWN
}

IN_GAME_ACTIONS: List[Direction] = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]
RELATIVE_ACTIONS: List[DirectionRelative] = [DirectionRelative.STRAIGHT, DirectionRelative.LEFT, DirectionRelative.RIGHT]

@dataclass
class GameState:
    apple_front: bool
    apple_left: bool
    apple_right: bool
    apple_back: bool
    danger_front: bool
    danger_left: bool
    danger_right: bool
    danger_front_left: bool
    danger_front_right: bool
    danger_back_left: bool
    danger_back_right: bool
    danger_front_2: bool
    danger_left_2: bool
    danger_right_2: bool
    moving_up: bool
    moving_left: bool
    moving_right: bool
    moving_down: bool
    
    def to_array(self) -> np.ndarray:
        # values for dx and dy are normalized
        return np.array([
            int(self.apple_front), int(self.apple_left),int(self.apple_right), int(self.apple_back),
            int(self.danger_front), int(self.danger_left), int(self.danger_right), 
            int(self.danger_front_left), int(self.danger_front_right), int(self.danger_back_left), int(self.danger_back_right),
            int(self.danger_front_2), int(self.danger_left_2), int(self.danger_right_2),
            int(self.moving_up), int(self.moving_left), int(self.moving_right), int(self.moving_down)
        ], dtype=float)
    

class SnakeGame:
    def __init__(self):
        self.apple: Position = (np.random.randint(0, GRID_HEIGHT), np.random.randint(0, GRID_WIDTH)) 
        self.snake: List[Position] = [(GRID_HEIGHT // 2, GRID_WIDTH // 2)]
        self.score: int = 0
        self.prev_direction: Direction = Direction.UP
        self.prev_head: Position | None = None
        self.snake_just_eat: bool = False
        self.first_move: bool = True
    
    def get_grid(self) -> np.ndarray:
        grid: np.ndarray = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)   # TODO: Change np zeros
        grid[*self.apple] = GameObject.APPLE.value
            
        for y, x in self.snake:
            grid[y, x] = GameObject.SNAKE.value
            
        return grid 
    
    def spawn_apple(self) -> None:
        apple_x: int = np.random.randint(0, GRID_WIDTH)
        apple_y: int = np.random.randint(0, GRID_HEIGHT)
        
        # Check spawnability
        while (apple_y, apple_x) in self.snake:
            apple_x = np.random.randint(0, GRID_WIDTH)
            apple_y = np.random.randint(0, GRID_HEIGHT)
            
        self.apple = (apple_y, apple_x)
        
    def increment_score(self) -> None:
        self.score += 1
    
    def is_apple_eaten(self) -> bool:
        return self.apple == self.snake[0]
    
    def is_snake_colliding(self) -> bool:
        # Wall collision
        head_y, head_x = self.snake[0]
        if head_y < 0 or head_y >= GRID_HEIGHT:
            return True
        if head_x < 0 or head_x >= GRID_WIDTH:
            return True
        
        if self.snake_just_eat:
            self.snake_just_eat = False
            return False
        
        # Head collision with body
        if self.snake[0] in self.snake[1:]:
            return True
        
        return False
    
    def add_snake_part(self) -> None:
        # Duplicate last position
        self.snake.append(self.snake[-1])
        self.snake_just_eat = True
        pass
    
    def move_snake_relative(self, rel_dir: DirectionRelative) -> None:
        if self.first_move:
            self.prev_direction = Direction.UP
            self.first_move = False
        
        match rel_dir:
            case DirectionRelative.STRAIGHT:
                self.move_snake(self.prev_direction)
            case DirectionRelative.LEFT:
                self.move_snake(LEFT_DIRECTIONS[self.prev_direction])
            case DirectionRelative.RIGHT:
                self.move_snake(RIGHT_DIRECTIONS[self.prev_direction])
            case _:
                return
        
    
    def move_snake(self, direction: Direction) -> None:
        if self.first_move:
            self.prev_direction = direction
            self.first_move = False
        
        # Check previous direction is not opposite
        if direction == OPPOSITE_DIRECTIONS.get(self.prev_direction):
            direction = self.prev_direction
        
        self.prev_head = self.snake[0]
        head_y, head_x = self.snake[0]
        
        match direction:
            case Direction.UP:
                new_head = (head_y - 1, head_x)
            case Direction.LEFT: 
                new_head = (head_y, head_x - 1)
            case Direction.RIGHT:
                new_head = (head_y, head_x + 1)
            case Direction.DOWN:
                new_head = (head_y + 1, head_x)
            case _:
                return
            
        self.snake.insert(0, new_head)
        self.snake.pop()
        self.prev_direction = direction
        
    def is_win(self) -> bool:
        return len(self.snake) == GRID_WIDTH * GRID_HEIGHT
    
    def get_state(self) -> GameState:
        head_y, head_x = self.snake[0]
        apple_y, apple_x = self.apple
        
        up = (head_y - 1, head_x)
        left = (head_y, head_x - 1)
        right = (head_y, head_x + 1)
        down = (head_y + 1, head_x)
        
        up_2 = (head_y - 2, head_x)
        left_2 = (head_y, head_x - 2)
        right_2 = (head_y, head_x + 2)
        down_2 = (head_y + 2, head_x)
        
        up_left = (head_y - 1, head_x - 1)
        up_right = (head_y - 1, head_x + 1)
        down_left = (head_y + 1, head_x - 1)
        down_right = (head_y + 1, head_x + 1)
        
        dir_up = self.prev_direction == Direction.UP
        dir_left = self.prev_direction == Direction.LEFT
        dir_right = self.prev_direction == Direction.RIGHT
        dir_down = self.prev_direction == Direction.DOWN
        
        adx = apple_x - head_x
        ady = apple_y - head_y
        
        apple_front = (dir_up and ady < 0) or (dir_left and adx < 0) or (dir_right and adx > 0) or (dir_down and ady > 0)
        apple_left = (dir_up and adx < 0) or (dir_left and ady > 0) or (dir_right and ady < 0) or (dir_down and adx > 0)
        apple_right = (dir_up and adx > 0) or (dir_left and ady < 0) or (dir_right and ady > 0) or (dir_down and adx < 0)
        apple_back = (dir_up and ady > 0) or (dir_left and adx > 0) or (dir_right and adx < 0) or (dir_down and ady < 0)
        
        def is_danger(point: Position) -> bool:
            y, x = point
            
            if y < 0 or y >= GRID_HEIGHT:
                return True
            if x < 0 or x >= GRID_WIDTH:
                return True
            if point in self.snake[1:]:
                return True
            return False
        
        danger_front = (dir_up and is_danger(up)) or (dir_left and is_danger(left)) or (dir_right and is_danger(right)) or (dir_down and is_danger(down))
        danger_left = (dir_up and is_danger(left)) or (dir_left and is_danger(down)) or (dir_right and is_danger(up)) or (dir_down and is_danger(right))
        danger_right = (dir_up and is_danger(right)) or (dir_left and is_danger(up)) or (dir_right and is_danger(down)) or (dir_down and is_danger(left))
        
        danger_front_left = (dir_up and is_danger(up_left)) or (dir_left and is_danger(down_left)) or (dir_right and is_danger(up_right)) or (dir_down and is_danger(down_right))
        danger_front_right = (dir_up and is_danger(up_right)) or (dir_left and is_danger(up_left)) or (dir_right and is_danger(down_right)) or (dir_down and is_danger(down_left))
        
        danger_back_left = (dir_up and is_danger(down_left)) or (dir_left and is_danger(down_right)) or (dir_right and is_danger(up_left)) or (dir_down and is_danger(up_right))
        danger_back_right = (dir_up and is_danger(down_right)) or (dir_left and is_danger(up_right)) or (dir_right and is_danger(down_left)) or (dir_down and is_danger(up_left))
        
        danger_front_2 = (dir_up and is_danger(up_2)) or (dir_left and is_danger(left_2)) or (dir_right and is_danger(right_2)) or (dir_down and is_danger(down_2))
        danger_left_2 = (dir_up and is_danger(left_2)) or (dir_left and is_danger(down_2)) or (dir_right and is_danger(up_2)) or (dir_down and is_danger(right_2))
        danger_right_2 = (dir_up and is_danger(right_2)) or (dir_left and is_danger(up_2)) or (dir_right and is_danger(down_2)) or (dir_down and is_danger(left_2))
        
        state = GameState(
            apple_front= apple_front,
            apple_left= apple_left,
            apple_right= apple_right,
            apple_back= apple_back,
            danger_front = danger_front,
            danger_left = danger_left,
            danger_right = danger_right,
            danger_front_left = danger_front_left,
            danger_front_right = danger_front_right,
            danger_back_left = danger_back_left,
            danger_back_right = danger_back_right,
            danger_front_2 = danger_front_2,
            danger_left_2 = danger_left_2,
            danger_right_2 = danger_right_2,
            moving_up = dir_up,
            moving_left = dir_left,
            moving_right = dir_right,
            moving_down = dir_down,
        )
        
        return state


if __name__ == '__main__':
    game = SnakeGame()
    state = game.get_state()
    print(state)
    print(state.to_array())