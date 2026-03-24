import random
import math
from typing import Self, Iterator

type Number = float | int


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: Number, y: Number) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __add__(self, other: "Vec2 | Number") -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        if isinstance(other, (float, int)):
            return Vec2(self.x + other, self.y + other)
        raise TypeError("Other must be either a Vec2 or a number")

    __radd__ = __add__

    def __iadd__(self, other: "Vec2 | Number") -> Self:
        if isinstance(other, Vec2):
            self.x += other.x
            self.y += other.y
        elif isinstance(other, (float, int)):
            self.x += other
            self.y += other
        else:
            raise TypeError("Other must be either a Vec2 or a number")
        return self

    def __sub__(self, other: "Vec2 | Number") -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        if isinstance(other, (float, int)):
            return Vec2(self.x - other, self.y - other)
        raise TypeError("Other must be either a Vec2 or a number")

    # __rsub__ = __sub__

    def __isub__(self, other: "Vec2 | Number") -> Self:
        if isinstance(other, Vec2):
            self.x -= other.x
            self.y -= other.y
        elif isinstance(other, (float, int)):
            self.x -= other
            self.y -= other
        else:
            raise TypeError("Other must be either a Vec2 or a number")
        return self

    def __truediv__(self, other: Number) -> "Vec2":
        return Vec2(self.x / other, self.y / other)

    def __itruediv__(self, other: float) -> Self:
        self.x /= other
        self.y /= other
        return self

    def __mul__(self, other: "Vec2 | Number") -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        if isinstance(other, (float, int)):
            return Vec2(self.x * other, self.y * other)
        raise TypeError("Other must be either a Vec2 or a number")

    __rmul__ = __mul__

    def __imul__(self, other: "Vec2 | Number") -> Self:
        if isinstance(other, Vec2):
            self.x *= other.x
            self.y *= other.y
        elif isinstance(other, (float, int)):
            self.x *= other
            self.y *= other
        else:
            raise TypeError("Other must be either a Vec2 or a number")
        return self

    def __str__(self) -> str:
        return f"Vec2({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"

    def __iter__(self) -> Iterator:
        yield self.x
        yield self.y

    def __copy__(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def sign(self) -> "Vec2":
        """Returns the sign of each component of the vector"""
        x = 1.0 if self.x > 0.0 else -1.0 if self.x < 0.0 else 0.0
        y = 1.0 if self.y > 0.0 else -1.0 if self.y < 0.0 else 0.0
        return Vec2(x, y)

    def norm_sq(self) -> float:
        """Returns the squared vector norm"""
        return self.x*self.x + self.y*self.y

    def norm(self) -> float:
        """Returns the vector norm"""
        return math.hypot(self.x, self.y)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def distance(self, other: "Vec2") -> float:
        """Calculates the distance between two vectors"""
        return math.hypot(self.x - other.x, self.y - other.y)

    def normalized(self) -> "Vec2":
        """Returns the normalized vector"""
        if self.norm_sq() == 0.0:
            return self
        return self / self.norm()

    def rotate(self, angle: float) -> "Vec2":
        """Returns a rotated vector, angle in radians"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    @staticmethod
    def zero() -> "Vec2":
        """Returns the zero vector"""
        return Vec2(0.0, 0.0)

    @staticmethod
    def random() -> "Vec2":
        """Returns a random vector"""
        return Vec2(random.random(), random.random())


if __name__ == "__main__":
    a = Vec2(2, 2)
    b = Vec2(-5, 3)
    c = 8
    z = Vec2.zero()

    assert a + b == Vec2(-3, 5)
    assert a + b == b + a
    assert a + c == Vec2(10, 10)
    assert c + a == a + c
    assert z.x == 0 and z.y == 0
    assert a * c == c * a == Vec2(16, 16)
    assert b / 2.0 == Vec2(-2.5, 1.5)

    a += 5
    assert a == Vec2(7, 7)
    
    import copy
    f = Vec2(5, -8)
    g = copy.copy(f)
    g += 2
    assert(f == Vec2(5, -8)) # <- Assertion error while it should not, must use copy.copy

    print("test passed")
