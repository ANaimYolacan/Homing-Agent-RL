"""
config.py — Central configuration for the Homing Agent project.

Every tunable parameter lives here. This is your single source of truth.
When something feels "off" during training, this is where you come to adjust.

Design philosophy: Use a dataclass so all settings are:
  - Typed (you see what type each value should be)
  - Documented (every field has a comment)
  - Easy to override (just pass keyword args)
"""

from dataclasses import dataclass


@dataclass
class Config:
    """All project settings in one place."""

    # ========================
    #  ARENA (the play field)
    # ========================
    ARENA_WIDTH: float = 800.0    # Width of the 2D arena in "world units"
    ARENA_HEIGHT: float = 800.0   # Height of the 2D arena

    # ========================
    #  TARGET MODE
    # ========================
    # This controls what the agent is trying to do:
    #   "static"  — Target is a random fixed point. Agent must reach it.
    #   "moving"  — Target is a threat rocket heading toward a protected
    #               point. Agent must INTERCEPT the threat before it arrives.
    TARGET_MODE: str = "static"

    # ========================
    #  AGENT PHYSICS
    # ========================
    # These define how the interceptor rocket behaves.
    AGENT_MASS: float = 1.0           # kg — heavier = harder to accelerate
    AGENT_MOI: float = 0.1            # kg·m² — moment of inertia (resistance to rotation)
    DRAG_COEFFICIENT: float = 0.05    # Linear drag — slows the agent proportional to speed
    ANGULAR_DRAG_COEFFICIENT: float = 0.3  # Rotational drag — slows spinning

    # Control limits — how powerful the agent's actuators are
    MAX_THRUST: float = 3.0    # Newtons — max forward push
    MAX_TORQUE: float = 0.5    # N·m — max rotational push

    # ========================
    #  THREAT ROCKET (used when TARGET_MODE = "moving")
    # ========================
    # The threat is a rocket heading toward the protected target.
    # For now it flies in a straight line at constant speed.
    # Later we can add evasive maneuvers, guidance, etc.
    THREAT_SPEED: float = 8.0     # units/s — constant speed toward protected target
    THREAT_MASS: float = 1.0      # kg (used for RigidBody, though we override velocity)
    THREAT_MOI: float = 0.1       # kg·m²
    THREAT_DRAG: float = 0.01     # Very low — threat maintains speed
    THREAT_ANGULAR_DRAG: float = 0.1

    # ========================
    #  PROTECTED TARGET (used when TARGET_MODE = "moving")
    # ========================
    # The thing being defended. Threat loses if agent intercepts it.
    # Agent loses if threat reaches this position.
    PROTECTED_X: float = 400.0       # Default: center of arena
    PROTECTED_Y: float = 400.0
    PROTECTED_RADIUS: float = 30.0   # Threat "hits" if it gets this close

    # ========================
    #  ENVIRONMENT
    # ========================
    GRAVITY: float = 0.0          # m/s² downward — 0 means "space-like" (no gravity)
    SPAWN_MARGIN: float = 50.0    # Keep spawns this far from arena edges
    MIN_SPAWN_DISTANCE: float = 200.0  # Minimum starting distance (agent ↔ target)
    TARGET_RADIUS: float = 25.0   # Agent "arrives" / "intercepts" when this close

    # ========================
    #  SIMULATION TIMING
    # ========================
    DT: float = 0.05       # Seconds per physics step (20 Hz)
    MAX_STEPS: int = 500   # Max steps before episode is truncated (= 25 seconds of sim time)

    # ========================
    #  OBSERVATION NORMALIZATION
    # ========================
    MAX_VELOCITY: float = 15.0          # m/s — velocities beyond this get clipped to +/-1
    MAX_ANGULAR_VELOCITY: float = 5.0   # rad/s — angular velocity clip

    # ========================
    #  REWARD SHAPING
    # ========================
    # The reward function is the most important part of RL.
    # These weights control what the agent "cares about."
    REWARD_APPROACH_SCALE: float = 100.0  # How much to reward getting closer
    REWARD_HEADING_SCALE: float = 0.1     # How much to reward pointing at target
    REWARD_FUEL_PENALTY: float = 0.01     # Penalty per unit of thrust used
    REWARD_STEP_PENALTY: float = -0.05    # Small penalty each step (encourages speed)
    REWARD_ARRIVAL_BONUS: float = 100.0   # Big bonus for reaching target / intercepting threat
    REWARD_OOB_PENALTY: float = -50.0     # Big penalty for going out of bounds

    # ── Moving mode extras ──
    REWARD_DEFENSE_FAILURE: float = -100.0  # Penalty if threat reaches protected target
    REWARD_URGENCY_SCALE: float = 0.05      # Penalty that grows as threat nears protected target
