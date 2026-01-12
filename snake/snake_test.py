from pathlib import Path

import pygame as pg
import numpy as np

from constants import *
from snake_ai import SnakeAI
from snake_game import RELATIVE_ACTIONS
from snake_wrapper import SnakeGameWrapper
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
    game = SnakeGameWrapper()
    state = game.reset()
    accumulator: float = 0.0
    delta_time: float = 1.0 / 30.0
    steps = 0
    
    prev_state = state
    
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
            steps += 1
            
            prev_state = state
            direction = ai.get_action(state)
            state, _, terminated = game.step(direction)
            
            if terminated:
                running = False
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE} - Model: {model_name} - Score: {game.game.score}")
        alpha = accumulator / delta_time
        draw_game(screen, game.game, alpha)
        
        text_surface = font.render(f"Total steps {steps} | Steps without eating {game.steps_without_eating}", True, (255, 255, 255))
        screen.blit(text_surface, (0, 0))
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
        
    print(f"{model_name} final score: {game.game.score}")
    
    print("Last state before end")
    print(prev_state)
    
    print("Snake direction")
    print(game.game.direction)
    
    print("Action taken")
    print(RELATIVE_ACTIONS[ai.get_action(prev_state)])
    
    print("Final state")
    print(game.game.get_state())
      
    pg.quit()
    
    
if __name__ == '__main__':
    directory = Path("snake/models")
    models = list(directory.iterdir())
    for model in models:
        if model.suffix != ".npy":
            continue
        
        np.random.seed(42)
        best_model = np.load(model)
        best_ai = SnakeAI(24, 32, 3)
        best_ai.set_weights(best_model)
        test(best_ai, model.name)