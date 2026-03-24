import random

from steering.obstacle import Obstacle, RectObstacle, SphereObstacle
from steering.vec2 import Vec2
from steering.constants import LABYRINTH_WALL_SIZE

def create_world_labyrinth(size: Vec2) -> list[Obstacle]:
    hs = 0.5 * size
    wall_size = LABYRINTH_WALL_SIZE
    wall_up = RectObstacle(Vec2(0.0, -hs.y), Vec2(size.x, wall_size))
    wall_left = RectObstacle(Vec2(-hs.x, 0.0), Vec2(wall_size, size.y))
    wall_right = RectObstacle(Vec2(hs.x, 0.0), Vec2(wall_size, size.y))
    wall_down = RectObstacle(Vec2(0.0, hs.y), Vec2(size.x, wall_size))
    
    wall_1 = RectObstacle(Vec2(0.0, 0.0), Vec2(wall_size, 0.5 * size.y))
    wall_2 = RectObstacle(Vec2(0.0, 0.0), Vec2(0.5 * size.x, wall_size))
    
    wall_11 = RectObstacle(Vec2(-0.5 * hs.x, -0.5 * hs.y), Vec2(wall_size, 0.25 * size.y))
    wall_12 = RectObstacle(Vec2(-0.5 * hs.x, -0.5 * hs.y), Vec2(0.25 * size.x, wall_size))
    
    wall_21 = RectObstacle(Vec2(0.5 * hs.x, -0.5 * hs.y), Vec2(wall_size, 0.25 * size.y))
    wall_22 = RectObstacle(Vec2(0.5 * hs.x, -0.5 * hs.y), Vec2(0.25 * size.x, wall_size))
    
    wall_31 = RectObstacle(Vec2(-0.5 * hs.x, 0.5 * hs.y), Vec2(wall_size, 0.25 * size.y))
    wall_32 = RectObstacle(Vec2(-0.5 * hs.x, 0.5 * hs.y), Vec2(0.25 * size.x, wall_size))
    
    wall_41 = RectObstacle(Vec2(0.5 * hs.x, 0.5 * hs.y), Vec2(wall_size, 0.25 * size.y))
    wall_42 = RectObstacle(Vec2(0.5 * hs.x, 0.5 * hs.y), Vec2(0.25 * size.x, wall_size))
    
    return [wall_up, wall_left, wall_right, wall_down, wall_1, wall_2, wall_11, wall_12, wall_21, wall_22, wall_31, wall_32, wall_41, wall_42]


def create_world_random(size: Vec2, nb_obs: int) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    hs = 0.5 * size
    
    for _ in range(nb_obs):
        x = random.uniform(-hs.x, hs.x)
        y = random.uniform(-hs.y, hs.y)
        r = random.randint(1, 3)
        sphere_obs = SphereObstacle(Vec2(x, y), r)

        x = random.uniform(-hs.x, hs.x)
        y = random.uniform(-hs.y, hs.y)
        w = random.randint(1, 3)
        h = random.randint(1, 3)
        rect_obs = RectObstacle(Vec2(x, y), Vec2(w, h))
        
        obstacles.append(sphere_obs)
        obstacles.append(rect_obs)
        
    return obstacles