import math
import copy

from steering.agent import Agent
from steering.vec2 import Vec2
from steering.obstacle import Obstacle
from steering.constants import TARGET_RADIUS, DELTA_TIME, WORLD_SIZE, AGENT_INIT_POS, TARGET_POS, AGENT_DEFAULT_DIRECTION

import numpy as np


class World:
    __slots__ = ("size", "agent", "obstacles", "target_pos", "target_radius")

    def __init__(self) -> None:
        """Create a world of size (w, h) centered on (0, 0) with an agent and a target"""
        self.size = WORLD_SIZE
        self.agent = Agent(AGENT_INIT_POS)
        self.obstacles: list[Obstacle] = []
        self.target_pos = copy.copy(TARGET_POS)
        self.target_radius: float = TARGET_RADIUS

    @property
    def half_size(self) -> Vec2:
        return 0.5 * self.size

    def add_obstacle(self, obs: Obstacle) -> None:
        """Add an obstacle to the world"""
        self.obstacles.append(obs)

    def distance_to_target(self) -> float:
        """Returns distance between the agent and the target"""
        return (self.agent.position - self.target_pos).norm()

    def distance_to_obstacles(self) -> list[float]:
        """Returns a list of distances between the agent and obstacles if obstacles are detected"""
        origin = self.agent.position
        rays = self.agent.get_rays()

        distances: list[float] = []

        for direction in rays:
            min_dist = math.inf

            for obs in self.obstacles:
                t = obs.ray_intersection(origin, direction)

                if t >= 0 and t < min_dist:
                    min_dist = t

            if min_dist == math.inf or min_dist > self.agent.perception_radius:
                distances.append(self.agent.perception_radius)
            else:
                distances.append(min_dist)

        return distances

    def is_colliding(self) -> bool:
        """Check if the agent is colliding with an obstacle"""
        for obs in self.obstacles:
            if obs.dist(self.agent.position) <= self.agent.collision_radius:
                return True
        return False

    def is_target_reach(self) -> bool:
        """Check if the agent reached the target"""
        return self.distance_to_target() <= 2.0 * self.target_radius
    
    def move(self, action: int) -> None:
        if action == 0:
            self.agent.move_straight()
        elif action == 1:
            self.agent.move_left()
        elif action == 2:
            self.agent.move_right()
            
        self.agent.update(DELTA_TIME)
    
    def get_state(self) -> np.ndarray:
        to_target = self.target_pos - self.agent.position
        direction = self.agent.velocity.normalized()
        if direction == Vec2.zero():
            direction = copy.copy(AGENT_DEFAULT_DIRECTION)
        angle_to_target = math.atan2(direction.x * to_target.y - direction.y * to_target.x, direction.dot(to_target))
        
        arr = self.distance_to_obstacles() + [self.distance_to_target(), angle_to_target]
        return np.array(arr, dtype=np.float32)

if __name__ == "__main__":
    from steering.obstacle import SphereObstacle

    world = World()
    print(world.agent.get_rays())
    print(math.cos(math.pi / 6.0), math.sin(math.pi / 6.0))

    obs = SphereObstacle(Vec2(6, 0), 4.0)
    world.add_obstacle(obs)

    print(world.distance_to_obstacles())

    print(world.is_colliding())
    
    print(world.get_state())
