"""
vector2d.py — A minimal 2D vector class.

Why build this from scratch instead of using numpy?
Because you should understand every operation that happens in the physics.
Numpy is great (we'll use it in the RL parts), but for the physics engine,
transparency > performance. With only one rigid body, speed doesn't matter.

Coordinate system:
  - x increases to the RIGHT
  - y increases UPWARD (standard math convention)
  - Angles are measured counter-clockwise from the positive x-axis
"""

import math


class Vector2D:
    """A 2D vector with x and y components.

    Supports arithmetic operations (+, -, *, /), dot/cross products,
    magnitude, normalization, and rotation.

    Usage:
        v1 = Vector2D(3, 4)
        v2 = Vector2D(1, 0)
        v3 = v1 + v2            # Vector2D(4.000, 4.000)
        v1.magnitude()          # 5.0
        v1.normalized()         # Vector2D(0.600, 0.800)
        v2.rotate(math.pi / 2)  # Vector2D(0.000, 1.000)
    """

    # __slots__ tells Python to NOT create a __dict__ for each instance.
    # This saves memory and is slightly faster — a good habit for objects
    # that get created thousands of times (like vectors in a physics sim).
    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    # ──────────────────────────────────────────────
    #  Arithmetic operators
    # ──────────────────────────────────────────────

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector addition: v1 + v2"""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: 'Vector2D') -> 'Vector2D':
        """In-place addition: v1 += v2 (modifies v1)"""
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector subtraction: v1 - v2"""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __isub__(self, other: 'Vector2D') -> 'Vector2D':
        """In-place subtraction: v1 -= v2"""
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, scalar: float) -> 'Vector2D':
        """Scalar multiplication: v * 2.0"""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2D':
        """Reverse scalar multiplication: 2.0 * v"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector2D':
        """Scalar division: v / 2.0"""
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Vector2D':
        """Negation: -v"""
        return Vector2D(-self.x, -self.y)

    # ──────────────────────────────────────────────
    #  Vector operations
    # ──────────────────────────────────────────────

    def magnitude(self) -> float:
        """Length of the vector: |v| = sqrt(x² + y²)"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def magnitude_squared(self) -> float:
        """Squared length: x² + y². Avoids the sqrt — use when you only
        need to compare magnitudes (e.g., 'is this vector longer than that one?')."""
        return self.x ** 2 + self.y ** 2

    def normalized(self) -> 'Vector2D':
        """Unit vector (same direction, length 1).
        Returns zero vector if magnitude is near zero to avoid division by zero."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector2D(0.0, 0.0)
        return self / mag

    def dot(self, other: 'Vector2D') -> float:
        """Dot product: v1 · v2 = |v1| * |v2| * cos(angle between them).

        Key insight: if both vectors are unit vectors, the dot product
        tells you how aligned they are:
          +1 = same direction
           0 = perpendicular
          -1 = opposite directions

        We use this to check if the agent is pointing toward the target.
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Vector2D') -> float:
        """2D cross product (returns a scalar, not a vector).

        In 3D, cross product gives a vector perpendicular to both inputs.
        In 2D, that vector would point out of the screen, so we just get
        the z-component as a scalar.

        Positive = other is counter-clockwise from self.
        Negative = other is clockwise from self.
        """
        return self.x * other.y - self.y * other.x

    def rotate(self, angle_rad: float) -> 'Vector2D':
        """Rotate this vector by angle_rad radians (counter-clockwise).

        Uses the 2D rotation matrix:
            [cos θ  -sin θ] [x]
            [sin θ   cos θ] [y]

        This is the fundamental operation that connects angle → direction.
        For example, rotating Vector2D(1, 0) by π/2 gives Vector2D(0, 1).
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def angle(self) -> float:
        """Angle of this vector from the positive x-axis, in radians.
        Returns value in [-π, π].

        Uses atan2(y, x) which correctly handles all four quadrants,
        unlike atan(y/x) which can't distinguish (1,1) from (-1,-1).
        """
        return math.atan2(self.y, self.x)

    def distance_to(self, other: 'Vector2D') -> float:
        """Euclidean distance to another vector (treating both as points)."""
        return (other - self).magnitude()

    def copy(self) -> 'Vector2D':
        """Create an independent copy of this vector."""
        return Vector2D(self.x, self.y)

    # ──────────────────────────────────────────────
    #  Representation
    # ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, other: object) -> bool:
        """Approximate equality (within floating point tolerance)."""
        if not isinstance(other, Vector2D):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
