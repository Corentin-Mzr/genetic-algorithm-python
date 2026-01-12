from enum import Enum
from dataclasses import dataclass

import numpy as np

from constants import GRID_WIDTH, GRID_HEIGHT, Position

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

IN_GAME_ACTIONS: list[Direction] = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]
RELATIVE_ACTIONS: list[DirectionRelative] = [DirectionRelative.STRAIGHT, DirectionRelative.LEFT, DirectionRelative.RIGHT]

# [Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left]
# Relative to the snake direction
RELATIVE_VISION_DIRS: dict[Direction, list[Position]] = {
    Direction.UP:    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    Direction.DOWN:  [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    Direction.LEFT:  [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)],
    Direction.RIGHT: [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
}

Ray = tuple[float, float, float]

@dataclass
class GameState:
    # Ray = [inv_dist_wall, inv_dist_body, apple_found]
    # 8 x 3 = 24 input
    ray_front: Ray
    ray_front_right: Ray
    ray_right: Ray
    ray_back_right: Ray
    ray_back: Ray
    ray_back_left: Ray
    ray_left: Ray
    ray_front_left: Ray
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.ray_front, self.ray_front_right, 
            self.ray_right, self.ray_back_right, 
            self.ray_back, self.ray_back_left, 
            self.ray_left, self.ray_front_left,
        ], dtype=np.float32)
    
    

class SnakeGame:
    def __init__(self):
        self.snake: list[Position] = [(GRID_HEIGHT // 2, GRID_WIDTH // 2)]
        self.snake_set: set[Position] = set(self.snake)
        self.prev_head: Position | None = None
        self.just_ate: bool = False
        
        self.direction: Direction = Direction.UP
        self.apple: Position = self._spawn_apple()
        self.score: int = 0
    
    def get_grid(self) -> np.ndarray:
        grid: np.ndarray = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        grid[*self.apple] = GameObject.APPLE.value
            
        for y, x in self.snake:
            grid[y, x] = GameObject.SNAKE.value
            
        return grid 
    
    def _spawn_apple(self) -> Position:
        while True:
            x = np.random.randint(0, GRID_WIDTH)
            y = np.random.randint(0, GRID_HEIGHT)
            
            if (y, x) not in self.snake_set:
                return (y, x)
    
    def is_colliding(self) -> bool:
        y, x = self.snake[0]
        
        # Wall collision
        if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
            return True
        
        if self.just_ate:
            self.just_ate = False
            return False
        
        # Body collision
        return len([pos for pos in self.snake[1:] if pos == self.snake[0]]) > 0
    
    def _move(self) -> None:
        y, x = self.snake[0]
        
        match self.direction:
            case Direction.UP:
                new_head = (y - 1, x)
            case Direction.LEFT: 
                new_head = (y, x - 1)
            case Direction.RIGHT:
                new_head = (y, x + 1)
            case Direction.DOWN:
                new_head = (y + 1, x)
                
        self.prev_head = (y, x)   
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)
        
        # Apple eaten
        if new_head == self.apple:
            self.score += 1
            self.apple = self._spawn_apple()
            self.just_ate = True
        else:
            self.snake.pop()
            
        self.snake_set = set(self.snake)
        
    
    def move_relative(self, direction: DirectionRelative) -> None:
        if direction == DirectionRelative.LEFT:
            self.direction = LEFT_DIRECTIONS[self.direction]
        elif direction == DirectionRelative.RIGHT:
            self.direction = RIGHT_DIRECTIONS[self.direction]
                
        self._move()
        
    
    def move_absolute(self, direction: Direction) -> None:
        # Check previous direction is not opposite
        if direction != OPPOSITE_DIRECTIONS.get(self.direction):
            self.direction = direction
            
        self._move()
        
    def is_win(self) -> bool:
        return len(self.snake) == GRID_WIDTH * GRID_HEIGHT
    
    def get_state(self) -> GameState:
        
        def get_ray(direction: Position) -> Ray:
            y, x = self.snake[0]
            dy, dx = direction
            
            wall = 0
            body = 0
            apple = 0
            dist = 1
            
            body_found = False
            
            while True:
                y += dy
                x += dx
                
                # Wall
                if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
                    wall = 1 / dist
                    break
                    
                # Body
                if (y, x) in self.snake_set and (y, x) != self.snake[0] and not body_found:
                    body = 1 / dist
                    body_found = True
                    
                # Apple
                if (y, x) == self.apple:
                    apple = 1
                    
                dist += 1
                    
            return wall, body, apple
        
        
        directions = RELATIVE_VISION_DIRS[self.direction]
        rays = [get_ray(dir) for dir in directions]
        state = GameState(*rays)
        
        return state


if __name__ == '__main__':
    np.random.seed(42)
    
    game = SnakeGame()
    game.apple = (3, 5)
    print(f"Snake position {game.snake} | Apple position {game.apple}")
    state = game.get_state()
    print(state)
    print(state.to_array())
    
    # Test 1: Basic move
    game = SnakeGame()
    print("Init pos:", game.snake[0])
    game.move_relative(DirectionRelative.STRAIGHT)
    print("After one move:", game.snake[0])
    
    # Test 2: Apple eat
    game.apple = (game.snake[0][0] - 1, game.snake[0][1])
    initial_length = len(game.snake)
    game.move_relative(DirectionRelative.STRAIGHT)
    print(f"Length before: {initial_length}, after: {len(game.snake)}")
    assert len(game.snake) == initial_length + 1, "Snake should grow"
    
    # Test 3: Wall collision
    game = SnakeGame()
    game.snake_set.clear()
    game.snake = [(0, 5)]
    game.snake_set.add((0, 5))
    game.direction = Direction.UP
    game.move_relative(DirectionRelative.STRAIGHT)
    assert game.is_colliding(), "Should collide with wall"
    
    print("\nTESTS OK")