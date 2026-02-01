import pygame as pg

from game_2048.constants import *
from game_2048.game import Game2048

from math import log2

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def lerp_color(a: Color, b: Color, t: float) -> Color:
    return int(lerp(a[0], b[0], t)), int(lerp(a[1], b[1], t)), int(lerp(a[2], b[2], t))

def get_cell_color(v: int) -> Color:

    colors_rgb: dict[int, Color] = {
    0:    (205, 193, 180),
    2:    (238, 228, 218),
    4:    (237, 224, 200),
    8:    (242, 177, 121),
    16:   (245, 149, 99),
    32:   (246, 124, 95),
    64:   (246, 94, 59),
    128:  (237, 207, 114),
    256:  (237, 204, 97),
    512:  (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
    }
    
    c = colors_rgb.get(v)
    if c is not None:
        return c
    return lerp_color(colors_rgb[2048], BLACK, min(1.0, log2(v) / 16))

def clear(screen: pg.Surface, color: Color = BLACK) -> None:
    """ Clear the screen """
    screen.fill(color)
    
def draw_grid(screen: pg.Surface, width: int, height: int, color: Color = GREY) -> None:
    """ Draw the grid """
    cell_size_x = WINDOW_WIDTH // width
    cell_size_y = WINDOW_HEIGHT // height
    
    for y in range(0, WINDOW_HEIGHT, cell_size_y):
        for x in range(0, WINDOW_WIDTH, cell_size_x):
            rect = pg.Rect(x, y, cell_size_x, cell_size_y)
            pg.draw.rect(screen, color, rect, 1)
            
def draw_cells(screen: pg.Surface, grid: list[int], color: Color = BEIGE) -> None:
    """ Draw cells, color is given as base color """
    cell_size_x = WINDOW_WIDTH // GRID_SIZE
    cell_size_y = WINDOW_HEIGHT // GRID_SIZE
    font = pg.font.Font(None, FONT_SIZE)
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell = grid[y * GRID_SIZE + x]
            if cell != 0:
                
                # Cell
                rect = pg.Rect(x * cell_size_x, y * cell_size_y, cell_size_x, cell_size_y)
                pg.draw.rect(screen, get_cell_color(cell), rect)
                
                # Text
                font_size = max(16, FONT_SIZE - int(log2(cell)))
                font.set_point_size(font_size)
                text_surface = font.render(str(cell), True, WHITE)
                offset_x = 0.5 * text_surface.get_bounding_rect().w
                offset_y = 0.5 * text_surface.get_bounding_rect().h
                
                text_x = x * cell_size_y + 0.5 * cell_size_x - offset_x
                text_y = y * cell_size_x + 0.5 * cell_size_y - offset_y
                
                screen.blit(text_surface, (text_x, text_y))
                
def render(screen: pg.Surface, game: Game2048) -> None:
    clear(screen)
    draw_cells(screen, game.grid)
    draw_grid(screen, game.size, game.size)
