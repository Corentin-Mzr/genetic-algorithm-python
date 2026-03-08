import pygame as pg

from cartpole.env import Cartpole, Action, ACTIONS
from cartpole.constants import *
from cartpole.render import clear, draw
from cartpole.wrapper import CartpoleWrapper
from cartpole.ai import CartpoleAI

import numpy as np

def main() -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    running = True
    dt = 0
    
    # Game setup
    env = Cartpole()
    action: Action | None = None
    accumulator: float = 0.0
    
    # AI
    best_model = np.load("cartpole/models/best_ai.npy")
    best_ai = CartpoleAI(CartpoleWrapper().input_size, 4, CartpoleWrapper().output_size)
    best_ai.set_weights(best_model)
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
        # ZQSD - OKLM - ARROWS
        keys = pg.key.get_pressed()
        if keys[pg.K_q] or keys[pg.K_k] or keys[pg.K_LEFT]:
            action = Action.LEFT
        elif keys[pg.K_d] or keys[pg.K_m] or keys[pg.K_RIGHT]:
            action = Action.RIGHT
        else:
            action = Action.IDLE
            
        accumulator += dt
        
        # Update game
        while accumulator >= DELTA_TIME:
            accumulator -= DELTA_TIME
            
            # if action is not None:
            #     env.move(action)
                
            state = env.get_state()
            ai_action_idx = best_ai.get_action(state)
            ai_action = ACTIONS[ai_action_idx]
            env.move(ai_action)
                    
            if not running:
                break
            
        print(env.get_state())
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE}")
        
        clear(screen)
        draw(screen, env.x, env.theta)
        
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
      
    pg.quit()

if __name__ == '__main__':
    main()