"""
physics/ — A from-scratch 2D physics engine.

This package provides:
  - Vector2D: 2D vector math (addition, rotation, dot product, etc.)
  - RigidBody: A body with mass, position, velocity, angle, and forces
  - World: The simulation container that ties everything together
"""

from physics.vector2d import Vector2D
from physics.rigid_body import RigidBody
from physics.world import World
