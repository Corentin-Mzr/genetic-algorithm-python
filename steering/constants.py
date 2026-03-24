import math

from steering.vec2 import Vec2

AGENT_VELOCITY_MAX: float = 10.0
AGENT_STEERING_MAX: float = 10.0
AGENT_COLLISION_RADIUS: float = 1.0
AGENT_RAY_COUNT: int = 6
AGENT_FOV: float = 90.0 * math.pi / 180.0
AGENT_PERCEPTION_RADIUS: float = 4.0
AGENT_DEFAULT_DIRECTION: Vec2 = Vec2(1.0, 0.0)
AGENT_LOCAL_VERTICES: list[Vec2] = [Vec2(1.0, 0.0), Vec2(-0.5, 0.5), Vec2(-0.5, -0.5)]
AGENT_INIT_POS: Vec2 = Vec2(-20.0, -20.0)

TARGET_RADIUS: float = 1.0
TARGET_POS: Vec2 = Vec2(20.0, 20.0)

DELTA_TIME: float = 1.0 / 60.0

WORLD_SIZE: Vec2 = Vec2(50.0, 50.0)

LABYRINTH_WALL_SIZE: float = 1.0

WINDOW_WIDTH: int = 600
WINDOW_HEIGHT: int = 600
WINDOW_TITLE: str = "Steering"