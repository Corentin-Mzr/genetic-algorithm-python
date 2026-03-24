from pathlib import Path
from copy import deepcopy, copy
from dataclasses import dataclass

import numpy as np
import pygame as pg

from steering.ai import SteeringAI
from steering.wrapper import SteeringWrapper
from steering.render import clear, draw_agent, draw_obstacle, draw_target, View
from steering.constants import *
from steering.world import World
from steering.world_types import create_world_labyrinth



def to_enum_str(v: int) -> str:
    if v == 0:
        return "STRAIGHT"
    if v == 1:
        return "LEFT"
    if v == 2:
        return "RIGHT"
    return "UNKNOWN"

@dataclass
class SteeringTestResult:
    model_name: str
    score: float
    positions: list[Vec2]
    actions: list[int]
    rewards: list[float]
    steps: int
    
def test(ai: SteeringAI, model_name: str, num_trials: int = 50) -> SteeringTestResult:
    wrapper = SteeringWrapper()
    best_score = -math.inf
    best_actions = []
    best_rewards = []
    best_positions = []
    steps = 0
    
    for _ in range(num_trials):
        state = wrapper.reset()
        terminated = False
        actions = []
        rewards = []
        i = 0
        total_reward = 0.0
        positions = []
        
        while not terminated:
            direction = ai.get_action(state)
            state, reward, terminated = wrapper.step(direction)
            actions.append(direction)
            rewards.append(reward)
            positions.append(copy(wrapper.env.agent.position))
            total_reward += reward
            i += 1
        
        if total_reward > best_score:
            best_score = total_reward
            best_actions = deepcopy(actions)
            best_rewards = deepcopy(rewards)
            best_positions = deepcopy(positions)
            steps = i
        
    # Return best performance
    result = SteeringTestResult(
        model_name=model_name,
        score=best_score,
        positions=best_positions,
        actions=best_actions,
        rewards=best_rewards,
        steps=steps
    )
    
    return result


def visualize_test_result(result: SteeringTestResult) -> None:
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
    delta_time: float = DELTA_TIME
    
    world = World()
    obstacles = create_world_labyrinth(WORLD_SIZE)
    for obs in obstacles:
        world.add_obstacle(obs)
    
    
    view = View(Vec2.zero(), size=WORLD_SIZE)
    
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
            
        # Simulation
        if step == 0:
            world = World()
            obstacles = create_world_labyrinth(WORLD_SIZE)
            for obs in obstacles:
                world.add_obstacle(obs)
        
        
        
        # Rendering
        clear(screen)
        
        for obs in world.obstacles:
            draw_obstacle(screen, view, obs, (0, 128, 0))

        draw_target(screen, view, world.target_pos, world.target_radius, (128, 128, 0))
        draw_agent(screen, view, world.agent, (0, 0, 128), debug=True)
        
        model = f"Model: {result.model_name}"
        curr_step = f"Step: {step + 1}/{result.steps}"
        curr_reward = f"Reward: {(result.rewards[step - 1]) if step > 0 else 0 :.0f}"
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
            world.move(result.actions[step])
            accumulator -= delta_time
            step += 1
            step %= result.steps
        
        dt = clock.tick(60) / 1000
      
    pg.quit()



if __name__ == '__main__':
    directory = Path("steering/models")
    models = list(directory.iterdir())
    
    for model in models:
        if model.suffix != ".npy":
            continue
        
        if model.name != "best_ai.npy":
            continue
        
        best_model = np.load(model)
        best_ai = SteeringAI(SteeringWrapper().input_size, 8, SteeringWrapper().output_size)
        best_ai.set_weights(best_model)
        
        result = test(best_ai, model.name, num_trials=10)
        
        visualize_test_result(result)