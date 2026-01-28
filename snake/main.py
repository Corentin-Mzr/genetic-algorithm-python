import pygame as pg

from snake.render import draw_game
from snake.game import SnakeGame, Direction
from snake.constants import *

def main() -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    running = True
    dt = 0
    
    # Game setup
    game: SnakeGame = SnakeGame()
    direction: Direction | None = None
    accumulator: float = 0.0
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
        # ZQSD - OKLM - ARROWS
        keys = pg.key.get_pressed()
        if keys[pg.K_z] or keys[pg.K_o] or keys[pg.K_UP]:
            direction = Direction.UP
        if keys[pg.K_s] or keys[pg.K_l] or keys[pg.K_DOWN]:
            direction = Direction.DOWN
        if keys[pg.K_q] or keys[pg.K_k] or keys[pg.K_LEFT]:
            direction = Direction.LEFT
        if keys[pg.K_d] or keys[pg.K_m] or keys[pg.K_RIGHT]:
            direction = Direction.RIGHT
            
        accumulator += dt
        
        # Update game
        while accumulator >= DELTA_TIME:
            accumulator -= DELTA_TIME
            
            if direction is not None:
                game.move_absolute(direction)
                
            if game.is_colliding() or game.is_win():
                running = False
                    
            if not running:
                break
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE} - Score: {game.score}")
        alpha = accumulator / DELTA_TIME
        screen.fill("black")
        draw_game(screen, game, alpha)
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
      
    if game.is_win():
        print("You win !")
    else:
        print("You lose !")
    
    print(f"Your final score: {game.score}")
      
    pg.quit()

if __name__ == '__main__':
    main()