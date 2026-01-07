import pygame as pg
import numpy as np

from enum import Enum
from typing import List, Tuple

# Window
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_TITLE: str = "Snake"

# Game
GRID_WIDTH: int = 16
GRID_HEIGHT: int = 9
DELTA_TIME: float = 1.0 / 10.0

# Grid rendering
CELL_SIZE_X: int = WINDOW_WIDTH // GRID_WIDTH
CELL_SIZE_Y: int = WINDOW_HEIGHT // GRID_HEIGHT


class GameObject(Enum):
    AIR = 0
    SNAKE = 1
    APPLE = 2

class Direction(Enum):
    NONE = 0
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4
    
OPPOSITE_DIRECTIONS: List[Tuple[Direction, Direction]] = [
    (Direction.UP, Direction.DOWN),
    (Direction.DOWN, Direction.UP),
    (Direction.LEFT, Direction.RIGHT),
    (Direction.RIGHT, Direction.LEFT)
]

class SnakeGame:
    apple: Tuple[int, int] = (np.random.randint(0, GRID_HEIGHT), np.random.randint(0, GRID_WIDTH)) 
    snake: List[Tuple[int, int]] = [(GRID_HEIGHT // 2, GRID_WIDTH // 2)]
    score: int = 0
    prev_direction: Direction = Direction.NONE
    
    snake_just_eat: bool = False
    
    def get_grid(self) -> np.ndarray:
        grid: np.ndarray = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)   # TODO: Change np zeros
        grid[*self.apple] = GameObject.APPLE.value
            
        for y, x in self.snake:
            grid[y % GRID_HEIGHT, x % GRID_WIDTH] = GameObject.SNAKE.value
            
        return grid 
    
    def spawn_apple(self) -> None:
        apple_x: int = np.random.randint(0, GRID_WIDTH)
        apple_y: int = np.random.randint(0, GRID_HEIGHT)
        
        # Check spawnability
        while (apple_y, apple_x) in self.snake:
            apple_x = np.random.randint(0, GRID_WIDTH)
            apple_y = np.random.randint(0, GRID_HEIGHT)
            
        self.apple = (apple_y, apple_x)
        print(self.apple)
        
    def increment_score(self) -> None:
        self.score += 1
    
    def is_apple_eaten(self) -> bool:
        return self.apple == self.snake[0]
    
    def is_snake_colliding(self) -> bool:
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
    
    def move_snake(self, direction: Direction) -> None:
        if direction == Direction.NONE:
            return
        
        # Check previous direction is not opposite
        if (direction, self.prev_direction) in OPPOSITE_DIRECTIONS:
            direction = self.prev_direction
        
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
            
        new_head = (new_head[0] % GRID_HEIGHT, new_head[1] % GRID_WIDTH)
            
        self.snake.insert(0, new_head)
        self.snake.pop()
        self.prev_direction = direction
       
       
def draw_game(screen: pg.Surface, grid: np.ndarray) -> None:
    color_map = np.array([
        [0, 0, 0],      # AIR
        [0, 255, 0],    # SNAKE
        [255, 0, 0]     # APPLE
    ])
    
    rgb_grid = color_map[grid]
    rgb_grid = np.repeat(np.repeat(rgb_grid, CELL_SIZE_X, axis=0), CELL_SIZE_Y, axis=1)
    
    surface = pg.surfarray.make_surface(rgb_grid.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def main() -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pg.display.set_caption(WINDOW_TITLE)
    clock = pg.time.Clock()
    running = True
    dt = 0
    
    # Game setup
    game: SnakeGame = SnakeGame()
    direction: Direction = Direction.NONE
    accumulator: float = 0.0
    
    print(game.apple)
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
        keys = pg.key.get_pressed()
        if keys[pg.K_z]:
            direction = Direction.UP
        if keys[pg.K_s]:
            direction = Direction.DOWN
        if keys[pg.K_q]:
            direction = Direction.LEFT
        if keys[pg.K_d]:
            direction = Direction.RIGHT
            
        accumulator += dt
        
        # Update game
        while accumulator >= DELTA_TIME:
            accumulator -= DELTA_TIME
            
            game.move_snake(direction)
        
            if game.is_apple_eaten():
                game.add_snake_part()
                game.increment_score()
                game.spawn_apple()  
                
            if game.is_snake_colliding():
                running = False
                    
            if not running:
                break
        
        # Rendering
        screen.fill("black")
        draw_game(screen, game.get_grid())
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
        
    pg.quit()

if __name__ == '__main__':
    main()