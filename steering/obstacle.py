from abc import ABC, abstractmethod
import math

from steering.vec2 import Vec2


class Obstacle(ABC):

    @property
    @abstractmethod
    def vertices(self) -> list[Vec2]:
        """List of vertices that makes the obstacle"""
        ...

    @abstractmethod
    def contains(self, p: Vec2) -> bool:
        """Returns true if the point P is contained in the obstacle"""
        ...

    @abstractmethod
    def dist(self, p: Vec2) -> float:
        """Returns the distance between the point P and the obstacle"""
        ...

    def get_nearest_vertex(self, p: Vec2) -> int:
        """Returns the index of the nearest vertex of point P"""
        min_i = 0
        min_dist = math.inf
        for i, v in enumerate(self.vertices):
            dist = (v - p).norm_sq()
            if dist < min_dist:
                min_i = i
                min_dist = dist

        return min_i

    @abstractmethod
    def ray_intersection(self, origin: Vec2, direction: Vec2) -> float:
        """Returns the distance of intersection with a ray. -1 if no intersection"""
        ...


class RectObstacle(Obstacle):
    def __init__(self, position: Vec2, size: Vec2) -> None:
        assert size.x > 0 and size.y > 0
        self.pos = position
        self.size = size

    @property
    def half_size(self) -> Vec2:
        return 0.5 * self.size

    @property
    def vertices(self) -> list[Vec2]:
        pos = self.pos
        hs = self.half_size
        v1 = Vec2(pos.x - hs.x, pos.y - hs.y)
        v2 = Vec2(pos.x - hs.x, pos.y + hs.y)
        v3 = Vec2(pos.x + hs.x, pos.y + hs.y)
        v4 = Vec2(pos.x + hs.x, pos.y - hs.y)
        return [v1, v2, v3, v4]

    def contains(self, p: Vec2) -> bool:
        pos = self.pos
        hs = self.half_size
        return (pos.x - hs.x) <= p.x <= (pos.x + hs.x) and (pos.y - hs.y) <= p.y <= (
            pos.y + hs.y
        )

    def dist(self, p: Vec2) -> float:
        pos = self.pos
        hs = self.half_size
        dx = max(abs(p.x - pos.x) - hs.x, 0.0)
        dy = max(abs(p.y - pos.y) - hs.y, 0.0)
        return (dx**2 + dy**2) ** 0.5

    def ray_intersection(self, origin: Vec2, direction: Vec2) -> float:
        pos = self.pos
        hs = self.half_size

        min_x = pos.x - hs.x
        min_y = pos.y - hs.y
        max_x = pos.x + hs.x
        max_y = pos.y + hs.y

        if direction.x != 0.0:
            tx1 = (min_x - origin.x) / direction.x
            tx2 = (max_x - origin.x) / direction.x
        elif min_x <= origin.x <= max_x:
            tx1, tx2 = -math.inf, math.inf
        else:
            return -1.0

        if direction.y != 0.0:
            ty1 = (min_y - origin.y) / direction.y
            ty2 = (max_y - origin.y) / direction.y
        elif min_y <= origin.y <= max_y:
            ty1, ty2 = -math.inf, math.inf
        else:
            return -1.0

        tmin = max(min(tx1, tx2), min(ty1, ty2))
        tmax = min(max(tx1, tx2), max(ty1, ty2))

        if tmax < 0 or tmin > tmax:
            return -1.0
        if tmin < 0:
            return 0.0
        return tmin


class SphereObstacle(Obstacle):
    def __init__(self, position: Vec2, radius: float) -> None:
        assert radius > 0
        self.pos = position
        self.r = radius

    @property
    def vertices(self) -> list[Vec2]:
        count = 36
        dtheta = 2.0 * math.pi / count
        return [
            Vec2(
                self.pos.x + self.r * math.cos(i * dtheta),
                self.pos.y + self.r * math.sin(i * dtheta),
            )
            for i in range(count)
        ]

    def contains(self, p: Vec2) -> bool:
        return (self.pos - p).norm_sq() < self.r**2

    def dist(self, p: Vec2) -> float:
        return max((self.pos - p).norm() - self.r, 0.0)

    def ray_intersection(self, origin: Vec2, direction: Vec2) -> float:
        oc = origin - self.pos
        a = direction.norm_sq()
        b = 2.0 * oc.dot(direction)
        c = oc.norm_sq() - self.r**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return -1.0
        t1 = (-b - math.sqrt(disc)) / (2.0 * a)
        t2 = (-b + math.sqrt(disc)) / (2.0 * a)

        if t2 < 0:
            return -1.0
        if t1 < 0:
            return 0.0
        return t1


if __name__ == "__main__":
    rect = RectObstacle(Vec2.zero(), Vec2(10, 10))

    assert not rect.contains(Vec2(-6, 0))
    assert rect.contains(Vec2(2, 2))
    assert not rect.contains(Vec2(6, 0))

    assert rect.dist(Vec2(-6, 0)) == 1.0
    assert rect.dist(Vec2(2, 2)) == 0.0
    assert rect.dist(Vec2(-10, -8)) == 34**0.5

    sphere = SphereObstacle(Vec2.zero(), 10)
    assert not sphere.contains(Vec2(10, 10))
    assert sphere.contains(Vec2(1, 1))

    assert sphere.dist(Vec2(10, 10)) == 200**0.5 - 10
    assert sphere.dist(Vec2(1, 1)) == 0.0
