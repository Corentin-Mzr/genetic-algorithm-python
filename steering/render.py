import math

from steering.vec2 import Vec2
from steering.agent import Agent
from steering.obstacle import Obstacle
from steering.constants import AGENT_LOCAL_VERTICES

import pygame as pg

def clear(screen: pg.Surface) -> None:
    screen.fill("gray")

def agent_local_to_world(position: Vec2, velocity: Vec2) -> list[Vec2]:
    angle = math.atan2(velocity.y, velocity.x)

    local = AGENT_LOCAL_VERTICES

    vertices: list[Vec2] = []

    for v in local:
        new_v = v.rotate(angle) + position
        vertices.append(new_v)

    return vertices


class View:
    __slots__ = ("center", "size", "rotation")

    def __init__(self, center: Vec2, size: Vec2, rotation: float = 0.0) -> None:
        self.center = center
        self.size = size
        self.rotation = rotation % (2.0 * math.pi)

    def world_to_screen(self, point: Vec2, screen: Vec2) -> Vec2:
        """Converts world coords to screen coords"""
        # Translate
        p = point - self.center

        # Rotate
        if self.rotation != 0:
            cos_r = math.cos(self.rotation)
            sin_r = math.sin(self.rotation)

            x = p.x
            y = p.y

            p.x = x * cos_r - y * sin_r
            p.y = x * sin_r + y * cos_r

        # Scale
        p.x *= screen.x / self.size.x
        p.y *= screen.y / self.size.y

        # Center
        p += 0.5 * screen

        return p

    def screen_to_world(self, point: Vec2, screen: Vec2) -> Vec2:
        """Converts screen coords to world coords"""
        p = point

        p -= 0.5 * screen

        p.x *= self.size.x / screen.x
        p.y *= self.size.y / screen.y

        if self.rotation != 0:
            cos_r = math.cos(self.rotation)
            sin_r = math.sin(self.rotation)

            x = p.x
            y = p.y

            p.x = x * cos_r - y * sin_r
            p.y = x * sin_r + y * cos_r

        p += self.center

        return p

    def transform_vertices(self, vertices: list[Vec2], screen: Vec2) -> list[Vec2]:
        return [self.world_to_screen(v, screen) for v in vertices]


def draw_agent(
    screen: pg.Surface, view: View, agent: Agent, color: tuple, debug: bool = False
) -> None:
    screen_size = Vec2(*screen.get_size())
    agent_vertices = [
        tuple(view.world_to_screen(v, screen_size))
        for v in agent_local_to_world(agent.position, agent.velocity)
    ]

    if debug:
        ratio = screen_size.x / view.size.x
        agent_center = tuple(view.world_to_screen(agent.position, screen_size))
        # direction_arrow = tuple(view.world_to_screen(agent.position + ratio * 0.5 * agent.velocity.normalized(), screen_size))

        # Direction
        # pg.draw.line(screen, "darkgreen", agent_center, direction_arrow)

        # Perception sphere
        pg.draw.circle(
            screen, "blue", agent_center, ratio * agent.perception_radius, width=1
        )

        # Collision sphere
        pg.draw.circle(
            screen, "red", agent_center, ratio * agent.collision_radius, width=1
        )

        # Rays
        for ray in agent.get_rays():
            ray_pos = tuple(
                view.world_to_screen(
                    agent.position + agent.perception_radius * ray, screen_size
                )
            )
            pg.draw.line(screen, "darkgreen", agent_center, ray_pos)

    pg.draw.polygon(screen, color, agent_vertices)


def draw_obstacle(screen: pg.Surface, view: View, obs: Obstacle, color: tuple) -> None:
    screen_size = Vec2(*screen.get_size())
    obs_vertices = [tuple(view.world_to_screen(v, screen_size)) for v in obs.vertices]
    pg.draw.polygon(screen, color, obs_vertices)


def draw_target(
    screen: pg.Surface, view: View, pos: Vec2, radius: float, color: tuple
) -> None:
    screen_size = Vec2(*screen.get_size())
    target_pos = tuple(view.world_to_screen(pos, screen_size))
    radius *= screen_size.x / view.size.x
    pg.draw.circle(screen, color, target_pos, radius)
