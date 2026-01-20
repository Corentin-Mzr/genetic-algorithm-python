import pygame as pg
import numpy as np

from snake_game import SnakeGame, GameObject
from constants import *

def clear(screen: pg.Surface) -> None:
    screen.fill(DARK_BLUE)


def draw_apple(screen: pg.Surface, game: SnakeGame) -> None:
    ay, ax = game.apple
    pg.draw.rect(
        screen,
        COLOR_APPLE,
        (ax * CELL_SIZE_X, ay * CELL_SIZE_Y, CELL_SIZE_X, CELL_SIZE_Y),
    )


def draw_snake(screen: pg.Surface, game: SnakeGame, alpha: float) -> None:
    snake_length: int = len(game.snake)
    
    for i, (y, x) in enumerate(game.snake):
        if i == 0 and game.prev_head:
            py, px = game.prev_head
            y = py + (y - py) * alpha
            x = px + (x - px) * alpha
            
        gradient_factor = i / max(snake_length - 1, 1)
        
        head_color = COLOR_SNAKE_HEAD  # Brightest color at head
        tail_color = (
            COLOR_SNAKE_HEAD[0] // 3,  # Darker red
            COLOR_SNAKE_HEAD[1] // 3,  # Darker green
            COLOR_SNAKE_HEAD[2] // 3   # Darker blue
        )
        
        # Interpolate between head and tail colors
        current_color: Color = (
            int(head_color[0] + (tail_color[0] - head_color[0]) * gradient_factor),
            int(head_color[1] + (tail_color[1] - head_color[1]) * gradient_factor),
            int(head_color[2] + (tail_color[2] - head_color[2]) * gradient_factor)
        )

        pg.draw.rect(
            screen,
            current_color,
            (x * CELL_SIZE_X, y * CELL_SIZE_Y, CELL_SIZE_X, CELL_SIZE_Y),
        )


def draw_grid(screen: pg.Surface):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE_X):
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE_Y):
            rect = pg.Rect(x, y, CELL_SIZE_X, CELL_SIZE_Y)
            pg.draw.rect(screen, GREY, rect, 1)

def draw_game(screen: pg.Surface, game: SnakeGame, alpha: float) -> None:
    clear(screen)
    
    draw_grid(screen)

    draw_apple(screen, game)

    draw_snake(screen, game, alpha)  

def draw_state(screen: pg.Surface, state: np.ndarray) -> None:
    clear(screen)
    
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell_value = state[y, x]
            if cell_value == GameObject.AIR.value:
                continue
            elif cell_value == GameObject.SNAKE_HEAD.value:
                color = COLOR_SNAKE_HEAD
            elif cell_value == GameObject.SNAKE_BODY.value:
                color = COLOR_SNAKE_BODY
            elif cell_value == GameObject.APPLE.value:
                color = COLOR_APPLE
            else:
                continue
            
            pg.draw.rect(
                screen,
                color,
                (x * CELL_SIZE_X, y * CELL_SIZE_Y, CELL_SIZE_X, CELL_SIZE_Y),
            )
            
    draw_grid(screen)