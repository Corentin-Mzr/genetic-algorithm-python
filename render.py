from snake_game import SnakeGame, Direction
from constants import *
import pygame as pg


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
            
            
        # Calculate gradient factor (0.0 at head, 1.0 at tail)
        gradient_factor = i / max(snake_length - 1, 1)
        
        # Create gradient color
        # You can adjust these base colors to your preference
        head_color = COLOR_SNAKE  # Brightest color at head
        tail_color = (
            COLOR_SNAKE[0] // 3,  # Darker red
            COLOR_SNAKE[1] // 3,  # Darker green
            COLOR_SNAKE[2] // 3   # Darker blue
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


def draw_danger_cells(screen: pg.Surface, game: SnakeGame) -> None:
    state = game.get_state()
    head_y, head_x = game.snake[0]

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

    danger_cells = []

    if game.prev_direction == Direction.UP:
        if state.danger_front:
            danger_cells.append(up)
        if state.danger_left:
            danger_cells.append(left)
        if state.danger_right:
            danger_cells.append(right)
        if state.danger_front_left:
            danger_cells.append(up_left)
        if state.danger_front_right:
            danger_cells.append(up_right)
        if state.danger_back_left:
            danger_cells.append(down_left)
        if state.danger_back_right:
            danger_cells.append(down_right)
        if state.danger_front_2:
            danger_cells.append(up_2)
        if state.danger_left_2:
            danger_cells.append(left_2)
        if state.danger_right_2:
            danger_cells.append(right_2)

    elif game.prev_direction == Direction.LEFT:
        if state.danger_front:
            danger_cells.append(left)
        if state.danger_left:
            danger_cells.append(down)
        if state.danger_right:
            danger_cells.append(up)
        if state.danger_front_left:
            danger_cells.append(down_left)
        if state.danger_front_right:
            danger_cells.append(up_left)
        if state.danger_back_left:
            danger_cells.append(down_right)
        if state.danger_back_right:
            danger_cells.append(up_right)
        if state.danger_front_2:
            danger_cells.append(left_2)
        if state.danger_left_2:
            danger_cells.append(down_2)
        if state.danger_right_2:
            danger_cells.append(up_2)

    elif game.prev_direction == Direction.RIGHT:
        if state.danger_front:
            danger_cells.append(right)
        if state.danger_left:
            danger_cells.append(up)
        if state.danger_right:
            danger_cells.append(down)
        if state.danger_front_left:
            danger_cells.append(up_right)
        if state.danger_front_right:
            danger_cells.append(down_right)
        if state.danger_back_left:
            danger_cells.append(up_left)
        if state.danger_back_right:
            danger_cells.append(down_left)
        if state.danger_front_2:
            danger_cells.append(right_2)
        if state.danger_left_2:
            danger_cells.append(up_2)
        if state.danger_right_2:
            danger_cells.append(down_2)

    elif game.prev_direction == Direction.DOWN:
        if state.danger_front:
            danger_cells.append(down)
        if state.danger_left:
            danger_cells.append(right)
        if state.danger_right:
            danger_cells.append(left)
        if state.danger_front_left:
            danger_cells.append(down_right)
        if state.danger_front_right:
            danger_cells.append(down_left)
        if state.danger_back_left:
            danger_cells.append(up_right)
        if state.danger_back_right:
            danger_cells.append(up_left)
        if state.danger_front_2:
            danger_cells.append(down_2)
        if state.danger_left_2:
            danger_cells.append(right_2)
        if state.danger_right_2:
            danger_cells.append(left_2)

    for y, x in danger_cells:
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
            pg.draw.rect(
                screen,
                YELLOW,  # yellow
                (x * CELL_SIZE_X, y * CELL_SIZE_Y, CELL_SIZE_X, CELL_SIZE_Y),
                2,  # outline so snake is still visible
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
    
    draw_danger_cells(screen, game)
    
   
