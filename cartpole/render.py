from math import cos, sin

import pygame as pg

from cartpole.constants import DARK_BLUE, GREY

def clear(screen: pg.Surface, color: tuple[int, int, int] = DARK_BLUE) -> None:
    """ Clear the screen """
    screen.fill(color)
    
def draw(screen: pg.Surface, x: float, theta: float) -> None:
    """ Draw the cart and pole """
    width, height = screen.get_size()
    CART_WIDTH = 0.2 * width
    CART_HEIGHT = 0.2 * height
    POLE_LENGTH = 100
    
    norm_x = (x / 10.0 + 0.5)
    remap_x = 0.25 + norm_x * 0.5
    
    cx: float = width * remap_x
    cy: float = 0.5 * height - 0.5 * CART_HEIGHT
    
    rect = pg.Rect(cx - 0.5 * CART_WIDTH, cy, CART_WIDTH, CART_HEIGHT)
    pg.draw.rect(screen, GREY, rect, 1)
    
    pivot = (cx, cy)
    tip = (
        int(pivot[0] + POLE_LENGTH * sin(theta)),
        int(pivot[1] - POLE_LENGTH * cos(theta)),
    )
    
    pg.draw.line(screen, GREY, pivot, tip, 1)

    # pole tip circle
    pg.draw.circle(screen, GREY, tip, 7)

    # pivot circle
    pg.draw.circle(screen, GREY, pivot, 6)
    
    
    pg.draw.line(screen, GREY, (0.25 * width, 0.5 * height), (0.75 * width, 0.5 * height))
    
    
if __name__ == '__main__':
    print("hello world")