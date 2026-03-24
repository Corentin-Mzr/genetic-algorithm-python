import random

import pygame as pg

from steering.vec2 import Vec2
from steering.world import World
from steering.render import View, draw_obstacle, draw_target, draw_agent
from steering.constants import AGENT_INIT_POS, TARGET_POS, WORLD_SIZE
from steering.world_types import create_world_labyrinth
    

def main() -> None:
    pg.init()
    screen = pg.display.set_mode((600, 600))
    screen_size = Vec2(*screen.get_size())
    clock = pg.time.Clock()
    running = True
    dt = 0.0
    elapsed = 0.0

    world = World()

    view = View(Vec2.zero(), size=WORLD_SIZE)

    turn_right = False
    turn_left = False
    
    obstacles = create_world_labyrinth(WORLD_SIZE)
    for obs in obstacles:
        world.add_obstacle(obs)

    direction = Vec2.zero()
    
    steps = 0

    while running:
        steps += 1
        
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        keys_pressed = pg.key.get_just_pressed()
        if keys_pressed[pg.K_m]:
            turn_right = True
        if keys_pressed[pg.K_k]:
            turn_left = True

        keys_released = pg.key.get_just_released()
        if keys_released[pg.K_m]:
            turn_right = False
        if keys_released[pg.K_k]:
            turn_left = False

        if turn_right:
            print("turn right")
            world.move(2)
        elif turn_left:
            print("turn left")
            world.move(1)
        else:
            world.move(0)

        # Physics
        mouse_screen = Vec2(*pg.mouse.get_pos())
        mouse_world = view.screen_to_world(mouse_screen, screen_size)

        if world.is_target_reach():
            print(f"Target reached after {elapsed:.1f}s ({steps} steps)")
        if world.is_colliding():
            print(f"Collision detected at {world.agent.position}")

        # Rendering
        screen.fill("gray")

        for obs in world.obstacles:
            draw_obstacle(screen, view, obs, (0, 128, 0))

        draw_target(screen, view, world.target_pos, world.target_radius, (128, 128, 0))
        draw_agent(screen, view, world.agent, (0, 0, 128), debug=True)

        pg.display.flip()

        dt = clock.tick(60) / 1000
        elapsed += dt

    pg.quit()


if __name__ == "__main__":
    main()
