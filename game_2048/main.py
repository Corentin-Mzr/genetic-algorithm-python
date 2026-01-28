import pygame as pg

from game_2048.game import Game2048, Direction
from game_2048.constants import *
from game_2048.render import render 

def main() -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    running = True
    
    # Game setup
    game = Game2048()
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            if event.type == pg.KEYDOWN:
                if event.key in (pg.K_z, pg.K_o, pg.K_UP):
                    game.move(Direction.UP)
                if event.key in (pg.K_s, pg.K_l, pg.K_DOWN):
                    game.move(Direction.DOWN)
                if event.key in (pg.K_q, pg.K_k, pg.K_LEFT):
                    game.move(Direction.LEFT)
                if event.key in (pg.K_d, pg.K_m, pg.K_RIGHT):
                    game.move(Direction.RIGHT)
                
        if game.is_game_over():
            running = False
                
        if not running:
            break
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE} - Score: {game.score}")
        screen.fill("black")
        render(screen, game)
        pg.display.flip()
        clock.tick(60)
    
    print(f"Your final score: {game.score}")
      
    pg.quit()

if __name__ == '__main__':
    main()