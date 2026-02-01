from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

import pygame as pg
import numpy as np

from snake.constants import *
from snake.ai import SnakeAI
from snake.wrapper import SnakeGameWrapper
from snake.game import DirectionRelative
from snake.render import draw_game, draw_state

def to_enum_str(v: int) -> str:
    if v == DirectionRelative.LEFT.value:
        return "LEFT"
    if v == DirectionRelative.STRAIGHT.value:
        return "STRAIGHT"
    if v == DirectionRelative.RIGHT.value:
        return "RIGHT"
    return "UNKNOWN"

@dataclass
class SnakeTestResult:
    model_name: str
    score: int
    grids: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    steps: int
    
def test(ai: SnakeAI, model_name: str, num_trials: int = 50) -> SnakeTestResult:
    env = SnakeGameWrapper()
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
        
        # Store initial state
        grids.append(env.game.get_grid())
        
        while not terminated:
            direction = ai.get_action(state)
            state, reward, terminated = env.step(direction)
            grids.append(env.game.get_grid())
            actions.append(direction)
            rewards.append(reward)
        
        if env.game.score > best_score:
            best_score = env.game.score
            best_grids = deepcopy(grids)
            best_actions = deepcopy(actions)
            best_rewards = deepcopy(rewards)
            steps = env.game.steps_total
        
    # Return best performance
    result = SnakeTestResult(
        model_name=model_name,
        score=best_score,
        grids=best_grids,
        actions=best_actions,
        rewards=best_rewards,
        steps=steps
    )
    
    return result

def visualize_test_result(result: SnakeTestResult) -> None:
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
    delta_time: float = 1.0 / 24.0
    
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
        draw_state(screen, result.grids[step])
        rendered_text = f"Model: {result.model_name} | Step: {step + 1}/{result.steps} | Reward: {result.rewards[step]:.0f} | Action: {to_enum_str(result.actions[step])} | Final score: {result.score}"
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
        

def test_with_render(ai: SnakeAI, model_name: str) -> None:
    # Pygame setup
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pg.time.Clock()
    running = True
    dt = 0
    
    # Text
    font = pg.font.Font(None, 24)
    
    # Game setup
    env = SnakeGameWrapper()
    state = env.reset()
    accumulator: float = 0.0
    delta_time: float = 1.0 / 15.0
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
            steps += 1
            
            direction = ai.get_action(state)
            state, _, terminated = env.step(direction)
            
            if terminated:
                running = False
        
        # Rendering
        pg.display.set_caption(f"{WINDOW_TITLE} - Model: {model_name} - Score: {env.game.score}")
        alpha = accumulator / delta_time
        draw_game(screen, env.game, alpha)
        
        text_surface = font.render(f"Total steps {steps} | Steps without eating {env.game.steps_without_eating}", True, (255, 255, 255))
        screen.blit(text_surface, (0, 0))
        
        pg.display.flip()
        dt = clock.tick(60) / 1000
        
    print(f"{model_name} final score: {env.game.score}")
      
    pg.quit()
    
    
if __name__ == '__main__':
    directory = Path("snake/models")
    models = list(directory.iterdir())
    
    for model in models:
        if model.suffix != ".npy":
            continue
        
        if model.name != "best_ai.npy":
            continue
        
        best_model = np.load(model)
        best_ai = SnakeAI(SnakeGameWrapper().input_size, 7, SnakeGameWrapper().output_size)
        best_ai.set_weights(best_model)
        
        result = test(best_ai, model.name)
        visualize_test_result(result)
