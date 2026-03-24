import copy

from steering.vec2 import Vec2
from steering.constants import (
    AGENT_VELOCITY_MAX,
    AGENT_STEERING_MAX,
    AGENT_COLLISION_RADIUS,
    AGENT_RAY_COUNT,
    AGENT_FOV,
    AGENT_PERCEPTION_RADIUS,
    AGENT_DEFAULT_DIRECTION,
)


class Agent:
    __slots__ = (
        "position",
        "velocity",
        "acceleration",
        "velocity_max",
        "steering_max",
        "collision_radius",
        "ray_count",
        "fov",
        "perception_radius",
    )

    def __init__(self, pos: Vec2, vel: Vec2 = Vec2.zero()) -> None:
        self.position = copy.copy(pos)
        self.velocity = copy.copy(vel)
        self.acceleration = Vec2.zero()

        self.velocity_max: float = AGENT_VELOCITY_MAX
        self.steering_max: float = AGENT_STEERING_MAX

        self.collision_radius: float = AGENT_COLLISION_RADIUS
        self.ray_count: int = AGENT_RAY_COUNT
        self.fov: float = AGENT_FOV
        self.perception_radius: float = AGENT_PERCEPTION_RADIUS

    def apply_force(self, force: Vec2) -> None:
        """Apply a force on the agent"""
        self.acceleration += force

    def update(self, dt: float) -> None:
        """Update the agent"""
        self.velocity += self.acceleration * dt
        if self.velocity.norm() > self.velocity_max:
            self.velocity = self.velocity_max * self.velocity.normalized()

        self.position += self.velocity * dt

        self.acceleration *= 0.0

    def steer(self, target_position: Vec2) -> None:
        """Make the agent steer in the given direction"""
        desired = target_position - self.position
        if desired.norm_sq() < 1e-8:
            return

        desired = self.velocity_max * desired.normalized()
        steer_force = desired - self.velocity

        if steer_force.norm_sq() > self.steering_max**2:
            steer_force = self.steering_max * steer_force.normalized()

        self.apply_force(steer_force)
        
    def move_right(self) -> None:
        """ Steer to the right """
        if self.velocity == Vec2.zero():
            vel = copy.copy(AGENT_DEFAULT_DIRECTION)
        else:
            vel = Vec2(-self.velocity.y, self.velocity.x).normalized()
        
        direction = self.position + vel
        self.steer(direction)
    
    def move_left(self) -> None:
        """ Steer to the left """
        if self.velocity == Vec2.zero():
            vel = copy.copy(AGENT_DEFAULT_DIRECTION)
        else:
            vel = Vec2(self.velocity.y, -self.velocity.x).normalized()
        
        direction = self.position + vel
        self.steer(direction)
        
    def move_straight(self) -> None:
        """ Dont steer """
        if self.velocity == Vec2.zero():
            vel = copy.copy(AGENT_DEFAULT_DIRECTION)
        else:
            vel = self.velocity.normalized()
        
        direction = self.position + vel
        self.steer(direction)

    def get_rays(self) -> list[Vec2]:
        """Returns unit perception rays"""
        rays: list[Vec2] = []

        direction = self.velocity.normalized()
        if direction == Vec2.zero():
            direction = copy.copy(AGENT_DEFAULT_DIRECTION)

        start_angle = -0.5 * self.fov if self.ray_count > 1 else 0
        dth = self.fov / (self.ray_count - 1) if self.ray_count > 1 else 0

        for i in range(self.ray_count):
            angle = start_angle + i * dth
            ray = direction.rotate(angle)
            rays.append(ray)

        return rays

    def __repr__(self) -> str:
        return f"Agent(pos={self.position}, vel={self.velocity})"


if __name__ == "__main__":
    agent = Agent(Vec2.zero())
    agent.steer(Vec2(1.0, 0.0))
    agent.update(1.0)
    print(agent)
    print(agent.get_rays())
