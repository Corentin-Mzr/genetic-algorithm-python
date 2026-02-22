from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pygame as pg

from cartpole.env import Action
from cartpole.ai import CartpoleAI
from cartpole.wrapper import CartpoleWrapper
from cartpole.render import clear, draw
from cartpole.constants import *


def to_enum_str(v: int) -> str:
    if v == Action.IDLE.value:
        return "IDLE"
    if v == Action.LEFT.value:
        return "LEFT"
    if v == Action.RIGHT.value:
        return "RIGHT"
    return "UNKNOWN"

@dataclass
class CartpoleTestResult:
    model_name: str
    score: float
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    steps: int
    
def test(ai: CartpoleAI, model_name: str, num_trials: int = 5) -> CartpoleTestResult:
    env = CartpoleWrapper()
    best_score = -1.0
    best_grids = []
    best_actions = []
    best_rewards = []
    steps = 0
    
    for _ in range(num_trials):
        state = env.reset()
        terminated = False
        states = []
        actions = []
        rewards = []
        i = 1
        score = 0.0
        
        # Store initial state
        states.append(env.cartpole.get_state().copy())
        
        while not terminated:
            i += 1
            action = ai.get_action(state)
            state, reward, terminated = env.step(action)
            states.append(env.cartpole.get_state().copy())
            actions.append(action)
            rewards.append(reward)
            score += reward
        
        if score > best_score:
            best_score = score
            best_grids = deepcopy(states)
            best_actions = deepcopy(actions)
            best_rewards = deepcopy(rewards)
            steps = i
        
    # Return best performance
    result = CartpoleTestResult(
        model_name=model_name,
        score=best_score,
        states=best_grids,
        actions=best_actions,
        rewards=best_rewards,
        steps=steps
    )
    
    return result


def visualize_test_result(result: CartpoleTestResult) -> None:
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
    delta_time: float = 1.0 / 60.0
    
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
        x, th = float(result.states[step][0]), float(result.states[step][2])
        clear(screen)
        draw(screen, x, th)
        
        model = f"Model: {result.model_name}"
        curr_step = f"Step: {step + 1}/{result.steps}"
        curr_reward = f"Reward: {(result.rewards[step - 1]) if step > 0 else 0 :.1f}"
        next_action = f"Next Action: {to_enum_str(result.actions[step]) if step < result.steps - 1 else "None"}"
        final_score = f"Final score: {result.score:.1f}"
            
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
    directory = Path("cartpole/models")
    models = list(directory.iterdir())
    
    for model in models:
        if model.suffix != ".npy":
            continue
        
        if model.name != "best_ai.npy":
            continue
        
        best_model = np.load(model)
        best_ai = CartpoleAI(CartpoleWrapper().input_size, 2, CartpoleWrapper().output_size)
        best_ai.set_weights(best_model)
        
        result = test(best_ai, model.name)
        
        visualize_test_result(result)