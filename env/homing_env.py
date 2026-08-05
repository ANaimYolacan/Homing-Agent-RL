"""
homing_env.py — Gymnasium environment for the homing agent.

This is the BRIDGE between our custom physics engine and the RL world.
It translates:
  - Physics state → Observation (what the agent "sees")
  - Agent action → Physics commands (what the agent "does")
  - Physics outcome → Reward (how well the agent did)

Supports two modes (controlled by config.TARGET_MODE):
  "static"  — Navigate to a fixed random point
  "moving"  — Intercept a threat rocket before it hits a protected target

═══════════════════════════════════════════════════════
  OBSERVATION SPACE (12 dimensions — always the same shape)
═══════════════════════════════════════════════════════

  Index │ Name          │ Meaning
  ──────┼───────────────┼──────────────────────────────────
    0   │ dx            │ Relative x to target (normalized)
    1   │ dy            │ Relative y to target (normalized)
    2   │ vx            │ Agent x-velocity (normalized)
    3   │ vy            │ Agent y-velocity (normalized)
    4   │ cos(θ)        │ Agent heading x-component
    5   │ sin(θ)        │ Agent heading y-component
    6   │ ω             │ Agent angular velocity (normalized)
    7   │ distance      │ Scalar distance to target (normalized)
    8   │ target_vx     │ Target x-velocity (0 in static mode)
    9   │ target_vy     │ Target y-velocity (0 in static mode)
   10   │ target_cos(θ) │ Target heading x (0 in static mode)
   11   │ target_sin(θ) │ Target heading y (0 in static mode)

  Why 12 dimensions even in static mode?
  The neural network needs a FIXED input size. By always including
  target velocity/heading (zeros in static mode), the same network
  architecture works for both modes. The network simply learns to
  ignore the zero inputs in static mode.

═══════════════════════════════════════════════════════
  ACTION SPACE (2 dimensions, continuous)
═══════════════════════════════════════════════════════

  Same in both modes:
    action[0] → Thrust: [-1,1] mapped to [0,1] via (a+1)/2
    action[1] → Rotation: [-1,1] directly

═══════════════════════════════════════════════════════
  REWARD FUNCTION
═══════════════════════════════════════════════════════

  Static mode:
    approach_reward + heading_bonus - fuel_penalty - step_penalty
    + arrival_bonus (terminal)  or  + oob_penalty (terminal)

  Moving mode (same as above, plus):
    - urgency_penalty: grows as threat approaches protected target
    + intercept_bonus (terminal)  or  + defense_failure_penalty (terminal)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from physics.world import World
from config import Config


class HomingEnv(gym.Env):
    """Gymnasium environment for the homing agent.

    Works in two modes:
      Static: Agent navigates to a random fixed target.
      Moving: Agent intercepts a threat rocket heading toward a protected target.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: Config = None, render_mode: str = None):
        super().__init__()

        self.config = config or Config()
        self.render_mode = render_mode

        # ── Create the physics world ──
        self.world = World(self.config)

        # ── Internal tracking ──
        self._prev_distance = 0.0  # For computing approach reward
        self._step_count = 0

        # ── Define the action space ──
        # Two continuous actions, both in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── Define the observation space ──
        # 12 continuous values (see module docstring for details)
        # Fixed size regardless of mode — zeros fill unused slots
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

    # ──────────────────────────────────────────────
    #  Core Gymnasium methods
    # ──────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment for a new episode.

        Both the agent and target get NEW random positions every episode.
        In moving mode, the threat spawns at a random edge heading toward
        the protected target.

        Returns:
            observation: 12D numpy array.
            info: Dictionary with extra information.
        """
        super().reset(seed=seed)

        # Reset physics world (handles both modes internally)
        self.world.reset(rng=self.np_random)

        # Initialize tracking
        self._prev_distance = self.world.distance_to_target
        self._step_count = 0

        return self._build_observation(), self._build_info()

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Returns:
            observation: Updated 12D observation.
            reward: Scalar reward.
            terminated: True if episode ended naturally.
            truncated: True if max steps exceeded.
            info: Dictionary with debugging data.
        """
        # ── Parse actions ──
        # Thrust: map [-1, 1] → [0, 1] so full network range is useful
        raw_thrust = float(action[0])
        thrust = float(np.clip((raw_thrust + 1.0) / 2.0, 0.0, 1.0))

        # Rotation: use directly, clipped to [-1, 1]
        rotation = float(np.clip(action[1], -1.0, 1.0))

        # ── Step the physics ──
        self.world.step(thrust, rotation, self.config.DT)
        self._step_count += 1

        # ── Compute reward ──
        reward = self._compute_reward(thrust)

        # ── Check termination conditions ──
        terminated = False
        truncated = False

        if self.world.has_reached_target():
            # SUCCESS — agent reached target / intercepted threat
            terminated = True
            reward += self.config.REWARD_ARRIVAL_BONUS

        elif self.world.threat_reached_protected():
            # FAILURE — threat hit the protected target (moving mode only)
            terminated = True
            reward += self.config.REWARD_DEFENSE_FAILURE

        elif self.world.is_out_of_bounds():
            # FAILURE — agent left the arena
            terminated = True
            reward += self.config.REWARD_OOB_PENALTY

        elif self._step_count >= self.config.MAX_STEPS:
            # TIMEOUT — ran out of time
            truncated = True

        # ── Update tracking for next step ──
        self._prev_distance = self.world.distance_to_target

        return self._build_observation(), reward, terminated, truncated, self._build_info()

    # ──────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """Convert the current physics state into a 12D observation vector.

        The first 8 values describe the agent and its relationship to the
        target. The last 4 describe the target's own motion (zeros if static).
        """
        agent = self.world.agent
        config = self.config
        max_distance = math.sqrt(config.ARENA_WIDTH ** 2 + config.ARENA_HEIGHT ** 2)

        # ── Agent → Target relationship (indices 0-7) ──

        # Relative position to target (normalized by arena diagonal)
        rel_pos = self.world.target_position - agent.position
        dx = rel_pos.x / max_distance
        dy = rel_pos.y / max_distance

        # Agent velocity (normalized and clipped)
        vx = float(np.clip(agent.velocity.x / config.MAX_VELOCITY, -1.0, 1.0))
        vy = float(np.clip(agent.velocity.y / config.MAX_VELOCITY, -1.0, 1.0))

        # Agent heading as unit vector (naturally in [-1, 1])
        cos_angle = math.cos(agent.angle)
        sin_angle = math.sin(agent.angle)

        # Agent angular velocity (normalized and clipped)
        ang_vel = float(np.clip(
            agent.angular_velocity / config.MAX_ANGULAR_VELOCITY, -1.0, 1.0
        ))

        # Distance to target (normalized, always in [0, ~1])
        distance = self.world.distance_to_target / max_distance

        # ── Target motion (indices 8-11) ──
        # In static mode, these are all zero → network learns to ignore them.
        # In moving mode, these tell the agent how the threat is moving.

        if config.TARGET_MODE == "moving":
            threat = self.world.threat
            target_vx = float(np.clip(
                threat.velocity.x / config.MAX_VELOCITY, -1.0, 1.0
            ))
            target_vy = float(np.clip(
                threat.velocity.y / config.MAX_VELOCITY, -1.0, 1.0
            ))
            target_cos = math.cos(threat.angle)
            target_sin = math.sin(threat.angle)
        else:
            target_vx = 0.0
            target_vy = 0.0
            target_cos = 0.0
            target_sin = 0.0

        return np.array([
            dx, dy, vx, vy, cos_angle, sin_angle, ang_vel, distance,
            target_vx, target_vy, target_cos, target_sin,
        ], dtype=np.float32)

    def _compute_reward(self, thrust_used: float) -> float:
        """Compute the reward for the current step.

        The reward function is the MOST IMPORTANT part of RL.
        It's designed with multiple components that guide the agent
        toward the desired behavior.
        """
        config = self.config
        max_distance = math.sqrt(config.ARENA_WIDTH ** 2 + config.ARENA_HEIGHT ** 2)
        current_distance = self.world.distance_to_target

        # ── 1. Approach reward (both modes) ──
        # Positive when agent gets closer to target.
        # In moving mode, the target (threat) is also moving, so the agent
        # must close the gap FASTER than the threat moves.
        distance_delta = self._prev_distance - current_distance
        approach_reward = (distance_delta / max_distance) * config.REWARD_APPROACH_SCALE

        # ── 2. Heading bonus (both modes) ──
        # Reward for pointing toward the target.
        direction = self.world.direction_to_target
        heading = self.world.agent.heading
        heading_alignment = heading.dot(direction)
        heading_reward = heading_alignment * config.REWARD_HEADING_SCALE

        # ── 3. Fuel penalty (both modes) ──
        fuel_penalty = -thrust_used * config.REWARD_FUEL_PENALTY

        # ── 4. Step penalty (both modes) ──
        step_penalty = config.REWARD_STEP_PENALTY

        total = approach_reward + heading_reward + fuel_penalty + step_penalty

        # ── 5. Urgency penalty (moving mode only) ──
        # As the threat gets closer to the protected target, urgency grows.
        # This teaches the agent that TIME MATTERS — dawdling is punished.
        if config.TARGET_MODE == "moving":
            threat_dist = self.world.threat_distance_to_protected
            # Urgency is high when threat is close (low distance)
            urgency = (1.0 - threat_dist / max_distance) * config.REWARD_URGENCY_SCALE
            total -= urgency

        return total

    def _build_info(self) -> dict:
        """Build the info dictionary for debugging/logging."""
        info = {
            "distance": self.world.distance_to_target,
            "speed": self.world.agent.speed,
            "angle": self.world.agent.angle,
            "steps": self._step_count,
            "time": self.world.time,
            "reached_target": self.world.has_reached_target(),
            "out_of_bounds": self.world.is_out_of_bounds(),
            "mode": self.config.TARGET_MODE,
        }

        # Moving mode extras
        if self.config.TARGET_MODE == "moving":
            info["threat_distance_to_protected"] = self.world.threat_distance_to_protected
            info["defense_failed"] = self.world.threat_reached_protected()
            info["threat_speed"] = self.world.threat.speed

        return info
