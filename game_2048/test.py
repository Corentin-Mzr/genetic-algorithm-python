from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

from game_2048.game import Direction
from game_2048.ai import Game2048AI
from game_2048.wrapper import Game2048Wrapper
from game_2048.render import clear, draw_grid, draw_cells
from game_2048.constants import *

import numpy as np
import pygame as pg

def to_enum_str(v: int) -> str:
    if v == Direction.UP.value:
        return "UP"
    if v == Direction.LEFT.value:
        return "LEFT"
    if v == Direction.RIGHT.value:
        return "RIGHT"
    if v == Direction.DOWN.value:
        return "DOWN"
    return "UNKNOWN"

@dataclass
class Game2048TestResult:
    model_name: str
    score: int
    grids: list[list[int]]
    actions: list[int]
    rewards: list[float]
    steps: int
    
def test(ai: Game2048AI, model_name: str, num_trials: int = 50) -> Game2048TestResult:
    env = Game2048Wrapper()
    best_score = -1
    best_grids = []
    best_actions = []
    best_rewards = []
    steps = 0
    
    for _ in range(num_trials):
        state = env.reset()
        terminated = False
        grids = []
        actions = []
        rewards = []
        i = 1
        
        # Store initial state
        grids.append(env.game.grid.copy())
        
        while not terminated:
            i += 1
            direction = ai.get_action(state)
            state, reward, terminated = env.step(direction)
            grids.append(env.game.grid.copy())
            actions.append(direction)
            rewards.append(reward)
        
        if env.game.score > best_score:
            best_score = env.game.score
            best_grids = deepcopy(grids)
            best_actions = deepcopy(actions)
            best_rewards = deepcopy(rewards)
            steps = i
        
    # Return best performance
    result = Game2048TestResult(
        model_name=model_name,
        score=best_score,
        grids=best_grids,
        actions=best_actions,
        rewards=best_rewards,
        steps=steps
    )
    
    return result


def visualize_test_result(result: Game2048TestResult) -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pg.display.set_caption(WINDOW_TITLE)
    clock = pg.time.Clock()
    running = True
    dt = 0
    pause = False
    
    # Text
    font = pg.font.Font(None, 20)
    
    # Steps
    step = 0
    accumulator: float = 0.0
    delta_time: float = 1.0 / 5.0
    
    # Main loop
    while running:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
        # Inputs
        pressed_keys = pg.key.get_just_pressed()
        released_keys = pg.key.get_just_released()
        if released_keys[pg.K_p]:
            pause = not pause
        if pressed_keys[pg.K_LEFT]:
            step -= 1
            step %= result.steps
        if pressed_keys[pg.K_RIGHT]:
            step += 1
            step %= result.steps
        if pressed_keys[pg.K_RETURN] or pressed_keys[pg.K_ESCAPE]:
            running = False
        
        # Rendering
        h, w = GRID_SIZE, GRID_SIZE
        clear(screen)
        draw_cells(screen, result.grids[step])
        draw_grid(screen, h, w)
        
        model = f"Model: {result.model_name}"
        curr_step = f"Step: {step + 1}/{result.steps}"
        curr_reward = f"Reward: {(result.rewards[step - 1]) if step > 0 else 0 :.0f}"
        next_action = f"Next Action: {to_enum_str(result.actions[step]) if step < result.steps - 1 else "None"}"
        final_score = f"Final score: {result.score}"
            
        rendered_text = f"{model} | {curr_step} | {curr_reward} | {next_action} | {final_score}"
        text_surface = font.render(rendered_text, True, (255, 255, 255))
        screen.blit(text_surface, (0, 0))
        pg.display.flip()
        
        if not pause:
            accumulator += dt
        
        # Update game
        while accumulator >= delta_time:
            accumulator -= delta_time
            step += 1
            step %= result.steps
        
        dt = clock.tick(60) / 1000
      
    pg.quit()



if __name__ == '__main__':
    directory = Path("game_2048/models")
    models = list(directory.iterdir())
    
    for model in models:
        if model.suffix != ".npy":
            continue
        
        if model.name != "best_ai.npy":
            continue
        
        best_model = np.load(model)
        best_ai = Game2048AI(Game2048Wrapper().input_size, 8, Game2048Wrapper().output_size)
        best_ai.set_weights(best_model)
        
        result = test(best_ai, model.name)
        
        visualize_test_result(result)