import pygame as pg
import numpy as np

from game_2048.constants import *
from game_2048.game import Game2048

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
            
def draw_cells(screen: pg.Surface, grid: np.ndarray, color: Color = BEIGE) -> None:
    """ Draw cells, color is given as base color """
    h, w = grid.shape
    
    cell_size_x = WINDOW_WIDTH // w
    cell_size_y = WINDOW_HEIGHT // h
    font = pg.font.Font(None, FONT_SIZE)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y, x]
            if cell != 0:
                
                # Cell
                f = (0.5 + 1.0 / cell)
                cell_color = (int(f * color[0]), int(0.5 * f * color[1]), int(0.5 * f * color[2]))
                rect = pg.Rect(x * cell_size_x, y * cell_size_y, cell_size_x, cell_size_y)
                pg.draw.rect(screen, cell_color, rect)
                
                # Text
                font_size = max(16, FONT_SIZE // int(len(str(cell)) ** 0.5))
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
    draw_grid(screen, game.w, game.h)