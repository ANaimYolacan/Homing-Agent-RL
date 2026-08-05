"""
test_env.py — Verify the physics engine and environment in both modes.

Run with:  python tests/test_env.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.vector2d import Vector2D
from physics.rigid_body import RigidBody
from physics.world import World
from config import Config


def test_vector2d():
    """Test basic vector operations."""
    print("=" * 60)
    print("TEST 1: Vector2D operations")
    print("=" * 60)

    v1 = Vector2D(3, 4)
    v2 = Vector2D(1, 0)

    v3 = v1 + v2
    assert v3 == Vector2D(4, 4), f"Addition failed: {v3}"
    print(f"  + {v1} + {v2} = {v3}")

    mag = v1.magnitude()
    assert abs(mag - 5.0) < 1e-10
    print(f"  + |{v1}| = {mag}")

    v_norm = v1.normalized()
    assert abs(v_norm.magnitude() - 1.0) < 1e-10
    print(f"  + normalize({v1}) = {v_norm}")

    dot = Vector2D(1, 0).dot(Vector2D(0, 1))
    assert abs(dot) < 1e-10
    print(f"  + (1,0) . (0,1) = {dot} (perpendicular)")

    v_rot = Vector2D(1, 0).rotate(math.pi / 2)
    assert abs(v_rot.x) < 1e-10 and abs(v_rot.y - 1.0) < 1e-10
    print(f"  + rotate((1,0), 90) = {v_rot}")

    print("  PASS: All Vector2D tests passed!\n")


def test_rigid_body():
    """Test physics simulation."""
    print("=" * 60)
    print("TEST 2: RigidBody physics")
    print("=" * 60)

    body = RigidBody(mass=1.0, drag_coeff=0.0, angular_drag_coeff=0.0)

    # F = ma
    body.reset(position=Vector2D(0, 0))
    body.apply_force(Vector2D(1.0, 0.0))
    body.step(dt=1.0)
    assert abs(body.velocity.x - 1.0) < 1e-6
    assert abs(body.position.x - 1.0) < 1e-6
    print("  + F=ma works (Newton's 2nd law)")

    # No force = constant velocity
    body.reset(position=Vector2D(0, 0), velocity=Vector2D(5, 0))
    body.step(dt=0.05)
    assert abs(body.velocity.x - 5.0) < 1e-6
    print("  + Inertia works (Newton's 1st law)")

    # Torque = angular acceleration
    body.reset(position=Vector2D(0, 0), angle=0.0)
    body.apply_torque(1.0)
    body.step(dt=0.1)
    assert body.angular_velocity > 0
    print("  + Torque -> rotation works")

    # Drag
    body_drag = RigidBody(mass=1.0, drag_coeff=0.5, angular_drag_coeff=0.0)
    body_drag.reset(velocity=Vector2D(10, 0))
    for _ in range(100):
        body_drag.step(dt=0.05)
    assert body_drag.speed < 10.0
    print(f"  + Drag works (10.0 -> {body_drag.speed:.4f})")

    print("  PASS: All RigidBody tests passed!\n")


def test_world_static():
    """Test world in static mode."""
    print("=" * 60)
    print("TEST 3: World — Static Mode")
    print("=" * 60)

    import numpy as np
    rng = np.random.default_rng(42)

    config = Config(TARGET_MODE="static")
    world = World(config)
    world.reset(rng=rng)

    print(f"  Agent: {world.agent.position}")
    print(f"  Target: {world.target_position}")
    print(f"  Distance: {world.distance_to_target:.1f}")

    assert world.distance_to_target >= config.MIN_SPAWN_DISTANCE
    print(f"  + Spawn distance OK (>= {config.MIN_SPAWN_DISTANCE})")

    assert not world.is_out_of_bounds()
    print("  + Agent in bounds")

    # Step with thrust
    for _ in range(10):
        world.step(thrust_fraction=1.0, rotation_fraction=0.0, dt=config.DT)
    assert world.agent.speed > 0
    print(f"  + After 10 steps: speed = {world.agent.speed:.3f}")

    print("  PASS: Static world works!\n")


def test_world_moving():
    """Test world in moving mode."""
    print("=" * 60)
    print("TEST 4: World — Moving Mode")
    print("=" * 60)

    import numpy as np
    rng = np.random.default_rng(42)

    config = Config(TARGET_MODE="moving")
    world = World(config)
    world.reset(rng=rng)

    print(f"  Agent:     {world.agent.position}")
    print(f"  Threat:    {world.threat.position}")
    print(f"  Protected: {world.protected_position}")
    print(f"  Threat -> Protected: {world.threat_distance_to_protected:.1f}")
    print(f"  Agent -> Threat:     {world.distance_to_target:.1f}")

    # Target position should track the threat
    assert world.target_position == world.threat.position
    print("  + target_position tracks threat")

    # Threat should be moving
    initial_threat_dist = world.threat_distance_to_protected
    for _ in range(50):
        world.step(thrust_fraction=0.0, rotation_fraction=0.0, dt=config.DT)

    new_threat_dist = world.threat_distance_to_protected
    assert new_threat_dist < initial_threat_dist, \
        f"Threat should get closer: {initial_threat_dist:.1f} -> {new_threat_dist:.1f}"
    print(f"  + Threat approaching: {initial_threat_dist:.1f} -> {new_threat_dist:.1f}")

    # Threat velocity should be roughly THREAT_SPEED
    threat_speed = world.threat.speed
    assert abs(threat_speed - config.THREAT_SPEED) < 1.0, \
        f"Threat speed {threat_speed:.1f} != {config.THREAT_SPEED}"
    print(f"  + Threat speed: {threat_speed:.1f} (expected {config.THREAT_SPEED})")

    print("  PASS: Moving world works!\n")


def test_env_static():
    """Test Gymnasium environment in static mode."""
    print("=" * 60)
    print("TEST 5: Gymnasium Env — Static Mode")
    print("=" * 60)

    import env as _env  # noqa: F401
    import gymnasium as gym

    environment = gym.make("HomingAgent-v0", config=Config(TARGET_MODE="static"))
    obs, info = environment.reset(seed=42)

    assert obs.shape == (12,), f"Expected (12,), got {obs.shape}"
    print(f"  + Observation shape: {obs.shape}")

    # Last 4 values should be zero in static mode
    assert all(obs[8:] == 0.0), f"Target motion should be 0 in static: {obs[8:]}"
    print(f"  + Target motion slots are zero: {obs[8:]}")

    total_reward = 0
    for step in range(100):
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  + Ran {step + 1} steps, total reward: {total_reward:.2f}")
    print(f"  + Final distance: {info['distance']:.1f}")

    # Reset should randomize
    obs2, info2 = environment.reset()
    assert info2['distance'] != info['distance'] or step < 10
    print(f"  + Reset randomizes (new dist: {info2['distance']:.1f})")

    environment.close()
    print("  PASS: Static env works!\n")


def test_env_moving():
    """Test Gymnasium environment in moving mode."""
    print("=" * 60)
    print("TEST 6: Gymnasium Env — Moving Mode")
    print("=" * 60)

    import env as _env  # noqa: F401
    import gymnasium as gym

    environment = gym.make("HomingAgent-v0", config=Config(TARGET_MODE="moving"))
    obs, info = environment.reset(seed=42)

    assert obs.shape == (12,), f"Expected (12,), got {obs.shape}"
    print(f"  + Observation shape: {obs.shape}")

    # Target motion slots should NOT all be zero (threat is moving)
    assert not all(obs[8:] == 0.0), f"Target motion should be non-zero: {obs[8:]}"
    print(f"  + Target motion slots populated: [{obs[8]:.3f}, {obs[9]:.3f}, {obs[10]:.3f}, {obs[11]:.3f}]")

    assert info['mode'] == 'moving'
    print(f"  + Mode: {info['mode']}")
    print(f"  + Threat -> protected: {info['threat_distance_to_protected']:.1f}")

    total_reward = 0
    defense_failed = False
    reached_target = False

    for step in range(500):
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward

        if terminated or truncated:
            defense_failed = info.get('defense_failed', False)
            reached_target = info.get('reached_target', False)
            break

    end_reason = "intercepted" if reached_target else \
                 "defense failed" if defense_failed else \
                 "OOB" if info.get('out_of_bounds') else "timeout"

    print(f"  + Ran {step + 1} steps, ended: {end_reason}")
    print(f"  + Total reward: {total_reward:.2f}")

    if defense_failed:
        print(f"  + Threat reached protected target (expected with random agent)")

    environment.close()
    print("  PASS: Moving env works!\n")


def test_observation_bounds():
    """Verify observation ranges for both modes."""
    print("=" * 60)
    print("TEST 7: Observation bounds (both modes)")
    print("=" * 60)

    import env as _env  # noqa: F401
    import gymnasium as gym
    import numpy as np

    labels = [
        "dx", "dy", "vx", "vy", "cos(a)", "sin(a)",
        "omega", "dist", "t_vx", "t_vy", "t_cos", "t_sin"
    ]

    for mode in ["static", "moving"]:
        obs_min = np.full(12, np.inf)
        obs_max = np.full(12, -np.inf)

        environment = gym.make("HomingAgent-v0", config=Config(TARGET_MODE=mode))

        for ep in range(5):
            obs, _ = environment.reset(seed=ep)
            obs_min = np.minimum(obs_min, obs)
            obs_max = np.maximum(obs_max, obs)
            for _ in range(200):
                action = environment.action_space.sample()
                obs, _, done, trunc, _ = environment.step(action)
                obs_min = np.minimum(obs_min, obs)
                obs_max = np.maximum(obs_max, obs)
                if done or trunc:
                    break

        print(f"\n  [{mode.upper()} MODE]")
        print(f"  {'Obs':>10s} | {'Min':>8s} | {'Max':>8s}")
        print(f"  {'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")
        for i, label in enumerate(labels):
            print(f"  {label:>10s} | {obs_min[i]:+8.4f} | {obs_max[i]:+8.4f}")

        environment.close()

    print("\n  PASS: Observation ranges look reasonable!\n")


if __name__ == "__main__":
    print("\n  HOMING AGENT - Environment Test Suite\n")

    test_vector2d()
    test_rigid_body()
    test_world_static()
    test_world_moving()
    test_env_static()
    test_env_moving()
    test_observation_bounds()

    print("=" * 60)
    print("  ALL TESTS PASSED - Environment is ready!")
    print("=" * 60)
