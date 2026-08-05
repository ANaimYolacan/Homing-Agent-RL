"""
world.py — The simulation world.

The World is the "arena" where everything happens. It supports two modes:

  STATIC MODE (TARGET_MODE = "static"):
    Agent must reach a randomly-placed fixed target.
    Simple navigation problem.

  MOVING MODE (TARGET_MODE = "moving"):
    A threat rocket is heading toward a protected target (e.g., a base).
    The agent must INTERCEPT the threat before it arrives.

    Scene layout:
      🛡️ Protected Target (static — the thing being defended)
            ↑
        🔴 Threat Rocket (flying toward protected target)
            ↑
        🔵 Agent (must intercept the threat)

    The threat currently flies in a straight line at constant speed.
    Future versions can add evasion, guidance, etc.
"""

import math
from physics.vector2d import Vector2D
from physics.rigid_body import RigidBody


class World:
    """2D simulation world containing an agent, a target, and optionally a threat.

    The World provides a unified interface regardless of mode:
      - world.target_position  → what the agent is chasing
      - world.distance_to_target → how far the agent is from it
      - world.has_reached_target() → did the agent arrive/intercept?

    In static mode, target_position is a fixed point.
    In moving mode, target_position is the threat rocket's position.
    """

    def __init__(self, config):
        self.config = config

        # ── The agent (interceptor) ──
        self.agent = RigidBody(
            mass=config.AGENT_MASS,
            moment_of_inertia=config.AGENT_MOI,
            drag_coeff=config.DRAG_COEFFICIENT,
            angular_drag_coeff=config.ANGULAR_DRAG_COEFFICIENT,
        )

        # ── The threat rocket (always exists, only MOVES in moving mode) ──
        self.threat = RigidBody(
            mass=config.THREAT_MASS,
            moment_of_inertia=config.THREAT_MOI,
            drag_coeff=config.THREAT_DRAG,
            angular_drag_coeff=config.THREAT_ANGULAR_DRAG,
        )

        # ── Static target (used in static mode) ──
        self._static_target = Vector2D(0.0, 0.0)

        # ── Protected target (used in moving mode) ──
        self.protected_position = Vector2D(config.PROTECTED_X, config.PROTECTED_Y)

        # ── Arena boundaries ──
        self.width = config.ARENA_WIDTH
        self.height = config.ARENA_HEIGHT

        # ── Simulation clock ──
        self.time = 0.0

    # ──────────────────────────────────────────────
    #  Unified target interface
    # ──────────────────────────────────────────────

    @property
    def target_position(self) -> Vector2D:
        """The position the agent is trying to reach.

        Static mode: the fixed target point.
        Moving mode: the threat rocket's CURRENT position (it's moving!).
        """
        if self.config.TARGET_MODE == "moving":
            return self.threat.position
        return self._static_target

    @property
    def distance_to_target(self) -> float:
        """Euclidean distance from the agent to whatever it's chasing."""
        return self.agent.position.distance_to(self.target_position)

    @property
    def direction_to_target(self) -> Vector2D:
        """Unit vector pointing from the agent toward its target."""
        diff = self.target_position - self.agent.position
        return diff.normalized()

    # ──────────────────────────────────────────────
    #  Threat-specific queries (moving mode)
    # ──────────────────────────────────────────────

    @property
    def threat_distance_to_protected(self) -> float:
        """How far the threat is from the protected target.
        Only meaningful in moving mode.
        """
        return self.threat.position.distance_to(self.protected_position)

    def threat_reached_protected(self) -> bool:
        """Has the threat reached the protected target?
        Only meaningful in moving mode. Always False in static mode.
        """
        if self.config.TARGET_MODE != "moving":
            return False
        return self.threat_distance_to_protected <= self.config.PROTECTED_RADIUS

    # ──────────────────────────────────────────────
    #  General queries
    # ──────────────────────────────────────────────

    def has_reached_target(self) -> bool:
        """Has the agent reached its target (static point or threat)?"""
        return self.distance_to_target <= self.config.TARGET_RADIUS

    def is_out_of_bounds(self) -> bool:
        """Check if the agent has left the arena."""
        p = self.agent.position
        return p.x < 0 or p.x > self.width or p.y < 0 or p.y > self.height

    # ──────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────

    def reset(self, rng) -> None:
        """Reset the world for a new episode.

        Randomizes positions based on the current mode.
        Called at the start of every training episode.

        Args:
            rng: A random number generator with a .uniform(low, high) method.
        """
        if self.config.TARGET_MODE == "moving":
            self._reset_moving(rng)
        else:
            self._reset_static(rng)

        self.time = 0.0

    def _reset_static(self, rng) -> None:
        """Reset for static mode: random agent + random fixed target."""
        margin = self.config.SPAWN_MARGIN

        # Random agent position and heading
        agent_x = rng.uniform(margin, self.width - margin)
        agent_y = rng.uniform(margin, self.height - margin)
        agent_angle = rng.uniform(-math.pi, math.pi)

        self.agent.reset(
            position=Vector2D(agent_x, agent_y),
            angle=agent_angle,
        )

        # Random target, enforcing minimum distance from agent
        for _ in range(100):
            target_x = rng.uniform(margin, self.width - margin)
            target_y = rng.uniform(margin, self.height - margin)
            candidate = Vector2D(target_x, target_y)

            if candidate.distance_to(self.agent.position) >= self.config.MIN_SPAWN_DISTANCE:
                self._static_target = candidate
                break
        else:
            self._static_target = candidate

        # Park the threat off-screen (not used in static mode)
        self.threat.reset(position=Vector2D(-100, -100))

    def _reset_moving(self, rng) -> None:
        """Reset for moving mode: threat at edge → protected target, agent random.

        The threat spawns near a random edge of the arena and heads straight
        toward the protected target. The agent spawns at a random position.
        """
        margin = self.config.SPAWN_MARGIN

        # ── Place the threat near a random edge ──
        edge = int(rng.integers(4)) if hasattr(rng, 'integers') else int(rng.randint(4))

        if edge == 0:    # Top edge
            tx = rng.uniform(margin, self.width - margin)
            ty = self.height - margin
        elif edge == 1:  # Right edge
            tx = self.width - margin
            ty = rng.uniform(margin, self.height - margin)
        elif edge == 2:  # Bottom edge
            tx = rng.uniform(margin, self.width - margin)
            ty = margin
        else:            # Left edge
            tx = margin
            ty = rng.uniform(margin, self.height - margin)

        threat_pos = Vector2D(float(tx), float(ty))

        # Threat heading: directly toward the protected target
        threat_direction = (self.protected_position - threat_pos).normalized()
        threat_angle = threat_direction.angle()

        self.threat.reset(
            position=threat_pos,
            velocity=threat_direction * self.config.THREAT_SPEED,
            angle=threat_angle,
        )

        # ── Place the agent at a random position ──
        # Ensure minimum distance from the threat (so agent has to work for it)
        for _ in range(100):
            ax = rng.uniform(margin, self.width - margin)
            ay = rng.uniform(margin, self.height - margin)
            agent_pos = Vector2D(float(ax), float(ay))

            if agent_pos.distance_to(threat_pos) >= self.config.MIN_SPAWN_DISTANCE:
                break

        agent_angle = rng.uniform(-math.pi, math.pi)
        self.agent.reset(
            position=agent_pos,
            angle=float(agent_angle),
        )

    # ──────────────────────────────────────────────
    #  Step
    # ──────────────────────────────────────────────

    def step(self, thrust_fraction: float, rotation_fraction: float, dt: float) -> None:
        """Step the simulation forward by dt seconds.

        Args:
            thrust_fraction: Engine power [0, 1]. Force applied along heading.
            rotation_fraction: Rotation command [-1, 1].
            dt: Time step in seconds.
        """
        # ── Agent physics (same in both modes) ──

        # Thrust along heading
        thrust_magnitude = thrust_fraction * self.config.MAX_THRUST
        thrust_force = self.agent.heading * thrust_magnitude
        self.agent.apply_force(thrust_force)

        # Rotation torque
        rotation_torque = rotation_fraction * self.config.MAX_TORQUE
        self.agent.apply_torque(rotation_torque)

        # Gravity (if any)
        if abs(self.config.GRAVITY) > 1e-10:
            gravity_force = Vector2D(0.0, -self.config.GRAVITY) * self.agent.mass
            self.agent.apply_force(gravity_force)

        # Step agent's rigid body
        self.agent.step(dt)

        # ── Threat movement (moving mode only) ──
        if self.config.TARGET_MODE == "moving":
            self._step_threat(dt)

        # ── Advance clock ──
        self.time += dt

    def _step_threat(self, dt: float) -> None:
        """Advance the threat rocket.

        Current behavior: constant-speed straight-line flight toward
        the protected target. The threat doesn't use force-based physics —
        we directly set its velocity each step for predictable behavior.

        Future enhancements could include:
          - Evasive maneuvers when the agent gets close
          - Proportional navigation guidance
          - Random course corrections
        """
        # Direction from threat to protected target
        direction = (self.protected_position - self.threat.position).normalized()

        # Set velocity directly (constant speed, always aimed at protected target)
        self.threat.velocity = direction * self.config.THREAT_SPEED

        # Update position
        self.threat.position += self.threat.velocity * dt

        # Update angle to match flight direction
        self.threat.angle = direction.angle()
