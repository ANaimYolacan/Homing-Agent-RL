"""
rigid_body.py — A 2D rigid body with Newtonian physics.

This is the physical "body" of our agent. It has:
  - Mass and moment of inertia (how hard it is to push/spin)
  - Position, velocity, angle, angular velocity (its current state)
  - Force/torque accumulators (we add forces, then step() applies them)

The key physics are two sets of parallel equations:

  LINEAR (translation):          ANGULAR (rotation):
  ─────────────────────          ─────────────────────
  F = m·a                       τ = I·α
  a = F / m                     α = τ / I
  v += a · dt                   ω += α · dt
  x += v · dt                   θ += ω · dt

Where:
  F = net force       τ = net torque
  m = mass            I = moment of inertia
  a = acceleration    α = angular acceleration
  v = velocity        ω = angular velocity
  x = position        θ = angle

Integration method: Semi-implicit Euler
  We update velocity FIRST, then use the NEW velocity to update position.
  This is more energy-stable than naive Euler (where you'd use old velocity).
  For our purposes, it's plenty accurate and dead simple to understand.
"""

import math
from physics.vector2d import Vector2D


class RigidBody:
    """A 2D rigid body that can be pushed and spun by forces and torques.

    Usage:
        body = RigidBody(mass=1.0)
        body.reset(position=Vector2D(100, 100))

        # Each simulation step:
        body.apply_force(Vector2D(1, 0))   # push it right
        body.apply_torque(0.5)              # spin it counter-clockwise
        body.step(dt=0.05)                  # advance physics by 0.05 seconds
    """

    def __init__(
        self,
        mass: float = 1.0,
        moment_of_inertia: float = 0.1,
        drag_coeff: float = 0.05,
        angular_drag_coeff: float = 0.3,
    ):
        # ── Physical properties (constant throughout simulation) ──

        self.mass = mass
        """kg — Resistance to linear acceleration. F = m·a, so higher mass
        means the same force produces less acceleration."""

        self.moment_of_inertia = moment_of_inertia
        """kg·m² — Resistance to angular acceleration. τ = I·α, same idea
        as mass but for rotation. Higher = harder to spin."""

        self.drag_coeff = drag_coeff
        """Drag coefficient for linear motion. Creates a force opposing
        velocity, proportional to speed. Prevents infinite acceleration."""

        self.angular_drag_coeff = angular_drag_coeff
        """Drag coefficient for rotation. Prevents infinite spinning."""

        # ── State variables (change every step) ──

        self.position = Vector2D(0.0, 0.0)
        """Current position in world coordinates."""

        self.velocity = Vector2D(0.0, 0.0)
        """Current velocity (direction + speed)."""

        self.angle = 0.0
        """Current heading in radians. 0 = pointing RIGHT (+x direction).
        Increases counter-clockwise (standard math convention)."""

        self.angular_velocity = 0.0
        """Current rate of rotation in rad/s. Positive = spinning CCW."""

        # ── Force accumulators (reset after each step) ──
        # We accumulate all forces/torques first, then apply them all at once.
        # This lets multiple systems add forces independently.

        self._net_force = Vector2D(0.0, 0.0)
        self._net_torque = 0.0

    # ──────────────────────────────────────────────
    #  Derived properties
    # ──────────────────────────────────────────────

    @property
    def heading(self) -> Vector2D:
        """Unit vector in the direction the body is currently facing.

        If angle = 0, heading = (1, 0) = pointing right.
        If angle = π/2, heading = (0, 1) = pointing up.

        This is what connects the agent's ANGLE to its THRUST DIRECTION.
        When the agent fires its engine, the force goes along this vector.
        """
        return Vector2D(math.cos(self.angle), math.sin(self.angle))

    @property
    def speed(self) -> float:
        """Scalar speed (magnitude of velocity)."""
        return self.velocity.magnitude()

    # ──────────────────────────────────────────────
    #  Force application
    # ──────────────────────────────────────────────

    def apply_force(self, force: Vector2D) -> None:
        """Add a force vector to the accumulator.

        Forces are NOT applied immediately — they're queued up and all
        applied together when step() is called. This is how real physics
        works: all forces act simultaneously, not sequentially.
        """
        self._net_force += force

    def apply_torque(self, torque: float) -> None:
        """Add a torque (rotational force) to the accumulator.

        Positive torque = counter-clockwise rotation.
        Negative torque = clockwise rotation.
        """
        self._net_torque += torque

    # ──────────────────────────────────────────────
    #  Physics step
    # ──────────────────────────────────────────────

    def step(self, dt: float) -> None:
        """Advance physics by dt seconds.

        This is where the magic happens. Every call to step():
        1. Calculates drag forces (opposing current motion)
        2. Computes acceleration from net force (F = ma → a = F/m)
        3. Updates velocity using acceleration
        4. Updates position using the NEW velocity (semi-implicit Euler)
        5. Does the same for angular dynamics
        6. Resets force accumulators for the next step
        """
        # ── Step 1: Add drag forces ──
        # Drag opposes velocity and is proportional to speed.
        # At high speeds, drag is strong. At low speeds, it's weak.
        # This naturally limits the agent's maximum speed.
        speed_sq = self.velocity.magnitude_squared()
        if speed_sq > 1e-10:  # avoid division by zero on stationary objects
            # drag = -c * |v| * v̂  (opposes direction of motion, scales with speed)
            drag_force = -self.drag_coeff * self.velocity.magnitude() * self.velocity.normalized()
            self._net_force += drag_force

        # Angular drag: same idea but for rotation
        angular_drag_torque = -self.angular_drag_coeff * self.angular_velocity
        self._net_torque += angular_drag_torque

        # ── Step 2: Linear dynamics (F = ma) ──
        acceleration = self._net_force / self.mass

        # ── Step 3: Update velocity FIRST (semi-implicit Euler) ──
        self.velocity += acceleration * dt

        # ── Step 4: Update position using NEW velocity ──
        # This is what makes it "semi-implicit" — we use the just-updated
        # velocity rather than the old one. It's a subtle difference but
        # makes the simulation much more stable for oscillatory systems.
        self.position += self.velocity * dt

        # ── Step 5: Angular dynamics (τ = Iα) ──
        angular_acceleration = self._net_torque / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        # Keep angle in [-π, π] — purely cosmetic, prevents unbounded growth
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        # ── Step 6: Reset accumulators ──
        # Forces are instantaneous — they only apply for one step.
        # If a force should persist (like gravity), it must be re-applied
        # every step by whoever is responsible for it.
        self._net_force = Vector2D(0.0, 0.0)
        self._net_torque = 0.0

    # ──────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────

    def reset(
        self,
        position: Vector2D = None,
        velocity: Vector2D = None,
        angle: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> None:
        """Reset the body to a given state. Clears all accumulated forces."""
        self.position = position.copy() if position else Vector2D(0.0, 0.0)
        self.velocity = velocity.copy() if velocity else Vector2D(0.0, 0.0)
        self.angle = angle
        self.angular_velocity = angular_velocity
        self._net_force = Vector2D(0.0, 0.0)
        self._net_torque = 0.0
