from enum import Enum
from dataclasses import dataclass

import numpy as np

from constants import GRID_WIDTH, GRID_HEIGHT, Position

class GameObject(Enum):
    AIR = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2
    APPLE = 3

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
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
    Direction.DOWN: Direction.UP,
}

LEFT_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.RIGHT: Direction.UP,
    Direction.DOWN: Direction.RIGHT,
}

RIGHT_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.RIGHT,
    Direction.LEFT: Direction.UP,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
}

ABSOLUTE_ACTIONS: list[Direction] = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]
RELATIVE_ACTIONS: list[DirectionRelative] = [DirectionRelative.STRAIGHT, DirectionRelative.LEFT, DirectionRelative.RIGHT]
VISION_DIRS: list[Position] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# [Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left]
# Relative to the snake direction
RELATIVE_VISION_DIRS: dict[Direction, list[Position]] = {
    Direction.UP:    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    Direction.LEFT:  [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)],
    Direction.RIGHT: [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
    Direction.DOWN:  [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
}

ABSOLUTE_TO_RELATIVE: dict[Direction, dict[DirectionRelative, Position]] = {
    Direction.UP: {
        DirectionRelative.STRAIGHT: (-1, 0), 
        DirectionRelative.LEFT: (0, -1), 
        DirectionRelative.RIGHT: (0, 1),
    },
    
    Direction.LEFT: {
        DirectionRelative.STRAIGHT: (0, -1), 
        DirectionRelative.LEFT: (1, 0), 
        DirectionRelative.RIGHT: (-1, 0),
    },
    
    Direction.RIGHT: {
        DirectionRelative.STRAIGHT: (0, 1), 
        DirectionRelative.LEFT: (-1, 0), 
        DirectionRelative.RIGHT: (1, 0),
    },
    
    Direction.DOWN: {
        DirectionRelative.STRAIGHT: (1, 0), 
        DirectionRelative.LEFT: (0, 1), 
        DirectionRelative.RIGHT: (0, -1),
    },
}

DIRECTION_TO_POSITION: dict[Direction, Position] = {
    Direction.UP: (-1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
    Direction.DOWN: (1, 0),
}

Ray = tuple[float, float, float]
PositionFloat = tuple[float, float]


@dataclass
class GameState:
    # Ray = [inv_dist_wall, inv_dist_body, inv_dist_apple]
    # 8 x 3 = 24 inputs
    ray_up: Ray
    ray_up_right: Ray
    ray_right: Ray
    ray_down_right: Ray
    ray_down: Ray
    ray_down_left: Ray
    ray_left: Ray
    ray_up_left: Ray
    
    # 2 inputs
    snake_head: PositionFloat
    
    # 2 inputs
    apple: PositionFloat
    
    def to_array(self) -> np.ndarray:
        rays = np.array([
            self.ray_up, self.ray_up_right, 
            self.ray_right, self.ray_down_right, 
            self.ray_down, self.ray_down_left, 
            self.ray_left, self.ray_up_left,], 
        dtype=np.float32).flatten()
        
        return np.concatenate(
            [rays, 
             np.array(self.snake_head, dtype=np.float32), 
             np.array(self.apple, dtype=np.float32)
        ]).flatten()
    
    

class SnakeGame:
    def __init__(self):
        self.snake: list[Position] = [self._spawn_snake()]
        self.snake_set: set[Position] = set(self.snake)
        self.prev_head: Position | None = None
        self.just_ate: bool = False
        self.first_apple_spawn: bool = True
        self.first_direction: bool = True
        
        self.direction: Direction = Direction.UP
        self.apple: Position = self._spawn_apple()
        self.score: int = 0
        
        self.steps_without_eating = 0
        self.steps_total = 0
    
    def get_grid(self) -> np.ndarray:
        """ Returns the game state as a grid"""
        grid: np.ndarray = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        grid[*self.apple] = GameObject.APPLE.value
        
        # Handle only valid snake positions    
        for y, x in self.snake:
            if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
                continue
            if (y, x) != self.snake[0]:
                grid[y, x] = GameObject.SNAKE_BODY.value
            else: 
                grid[y, x] = GameObject.SNAKE_HEAD.value
            
        return grid 
    
    def _spawn_snake(self) -> Position:
        """ Returns a random position for the initial snake head position """
        # Dont spawn close to walls
        x = np.random.randint(int(0.25 * GRID_WIDTH), int(0.75 * GRID_WIDTH))
        y = np.random.randint(int(0.25 * GRID_HEIGHT), int(0.75 * GRID_HEIGHT))
        return (y, x)
    
    def _spawn_apple(self) -> Position:
        """ Returns a random position for the apple"""
        if self.first_apple_spawn:
            min_f = 0.25
            max_f = 0.75
            self.first_apple_spawn = False
        else:
            min_f = 0
            max_f = 1.0
        
        while True:
            x = np.random.randint(int(min_f * GRID_WIDTH), int(max_f * GRID_WIDTH))
            y = np.random.randint(int(min_f * GRID_HEIGHT), int(max_f * GRID_HEIGHT))
            
            if (y, x) not in self.snake_set:
                return (y, x)
    
    def is_colliding(self) -> bool:
        """ Check if the snake is colliding with walls or itself """
        y, x = self.snake[0]
        
        # Wall collision
        if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
            return True
        
        # Dont check collision when snake just ate
        if self.just_ate:
            self.just_ate = False
            return False
        
        # Body collision
        return len([pos for pos in self.snake[1:] if pos == self.snake[0]]) > 0
    
    def _move(self) -> None:
        """ Handle snake movement and apple eating"""
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
        self.steps_without_eating += 1
        self.steps_total += 1
        
        # Apple eaten
        if new_head == self.apple:
            self.score += 1
            self.apple = self._spawn_apple()
            self.just_ate = True
            self.steps_without_eating = 0
        else:
            self.snake.pop()
            
        self.snake_set = set(self.snake)
        
    
    def move_relative(self, direction: DirectionRelative) -> None:
        """ Move using relative direction """
        if direction == DirectionRelative.LEFT:
            self.direction = LEFT_DIRECTIONS[self.direction]
        elif direction == DirectionRelative.RIGHT:
            self.direction = RIGHT_DIRECTIONS[self.direction]
                
        self._move()
        
    
    def move_absolute(self, direction: Direction) -> None:
        """ Move using absolute direction """
        if self.first_direction:
            self.direction = direction
            self.first_direction = False
        
        # Check previous direction is not opposite
        if direction != OPPOSITE_DIRECTIONS.get(self.direction):
            self.direction = direction
            
        self._move()
    
    def is_win(self) -> bool:
        """ Check if the snake has filled the grid """
        return len(self.snake) == GRID_WIDTH * GRID_HEIGHT
    
    def _get_ray(self, direction: Position) -> Ray:
        """ Get a ray in the given direction """
        y, x = self.snake[0]
        dy, dx = direction
        tail = self.snake[-1]
        
        wall = 0.0
        body = 0.0
        apple = 0.0
        dist = 1.0
        
        body_found = False
        
        while True:
            y += dy
            x += dx
            
            # Wall
            if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
                wall = 1.0 / dist
                break
                
            # Body
            is_body_part = (y, x) in self.snake_set
            if self.just_ate:
                if is_body_part and not body_found:
                    body = 1.0 / dist
                    body_found = True
            else:
                if is_body_part and (y, x) != tail and not body_found:
                    body = 1.0 / dist
                    body_found = True
                
            # Apple
            if (y, x) == self.apple:
                apple = 1.0 / dist
                
            dist += 1
                
        return wall, body, apple
    
    def get_state(self) -> GameState:
        """ Get the current game state """
        directions = RELATIVE_VISION_DIRS[self.direction]
        rays = [self._get_ray(dir) for dir in directions]
        snake_head = (self.snake[0][0] / GRID_HEIGHT, self.snake[0][1] / GRID_WIDTH)
        apple = (self.apple[0] / GRID_HEIGHT, self.apple[1] / GRID_WIDTH)
        state = GameState(*rays, snake_head=snake_head, apple=apple)
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
    game.move_absolute(Direction.UP)
    print("After one move:", game.snake[0])
    
    # Test 2: Apple eat
    game.apple = (game.snake[0][0] - 1, game.snake[0][1])
    initial_length = len(game.snake)
    game.move_absolute(Direction.UP)
    print(f"Length before: {initial_length}, after: {len(game.snake)}")
    assert len(game.snake) == initial_length + 1, "Snake should grow"
    
    # Test 3: Wall collision
    game = SnakeGame()
    game.snake_set.clear()
    game.snake = [(0, 5)]
    game.snake_set.add((0, 5))
    print(game.get_state().to_array())
    game.direction = Direction.UP
    game.move_absolute(Direction.UP)
    assert game.is_colliding(), "Should collide with wall"
    
    print("\nTESTS OK")
    
    
    
    # @dataclass
# class GameState:
#     # Ray = [inv_dist_wall, inv_dist_body, inv_dist_apple]
#     # 8 x 3 = 24 inputs
#     ray_up: Ray
#     ray_up_right: Ray
#     ray_right: Ray
#     ray_down_right: Ray
#     ray_down: Ray
#     ray_down_left: Ray
#     ray_left: Ray
#     ray_up_left: Ray
    
#     # 2 inputs
#     # apple: PositionFloat
    
#     # 1 input
#     apple: float
    
#     def to_array(self) -> np.ndarray:
#         return np.concatenate([
#             self.ray_up, self.ray_up_right, 
#             self.ray_right, self.ray_down_right, 
#             self.ray_down, self.ray_down_left, 
#             self.ray_left, self.ray_up_left,
#             [self.apple],
#         ], dtype=np.float32)
    
    
    
        # directions = RELATIVE_VISION_DIRS[self.direction]
        # rays = [get_ray(dir) for dir in directions]
        
        # sy, sx = self.snake[0]
        # ay, ax = self.apple
        # apple = ((ay - sy) / GRID_HEIGHT, (ax - sx) / GRID_WIDTH)
        # apple = (abs(ay - sy) + abs(ax - sx)) / (GRID_WIDTH + GRID_HEIGHT)
        
        # state = GameState(*rays, apple=apple)