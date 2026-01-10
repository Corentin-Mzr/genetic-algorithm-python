import pygame as pg
import numpy as np

from pathlib import Path

from constants import *
from snake_ai import SnakeAI
from snake_game import SnakeGame, Direction
from render import draw_game

def test(ai: SnakeAI, model_name: str) -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    running = True
    dt = 0
    
    # Text
    font = pg.font.Font(None, 24)
    
    # Game setup
    game: SnakeGame = SnakeGame()
    game.prev_direction = Direction.UP
    accumulator: float = 0.0
    delta_time: float = 1.0 / 30.0
    steps_no_eat = 0
    steps = 0
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            
        accumulator += dt
        
        # Update game
        while accumulator >= delta_time:
            accumulator -= delta_time
            steps_no_eat += 1
            steps += 1
            
            state = game.get_state()
            direction = ai.get_action(state.to_array())
            game.move_snake_relative(direction)
        
            if game.is_apple_eaten():
                game.add_snake_part()
                game.increment_score()
                game.spawn_apple()  
                steps_no_eat = 0
                
            if game.is_snake_colliding() or game.is_win() or steps_no_eat > 500:
                running = False
                    
            if not running:
                break
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE} - Model: {model_name} - Score: {game.score}")
        alpha = accumulator / delta_time
        screen.fill("black")
        draw_game(screen, game, alpha)
        
        text_surface = font.render(f"Total steps {steps} | Steps without eating {steps_no_eat}", True, (255, 255, 255))
        screen.blit(text_surface, (0, 0))
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
        
    print(f"{model_name} final score: {game.score}")
      
    pg.quit()
    
    
if __name__ == '__main__':
    directory = Path("models")
    models = list(directory.iterdir())
    for model in models:
        if model.suffix != ".npy":
            continue
        
        np.random.seed(42)
        best_model = np.load(model)
        best_ai = SnakeAI()
        best_ai.set_weights(best_model)
        test(best_ai, model.name)