"""
env/ — Gymnasium environment package.

Registers our custom environment so it can be created with:
    import gymnasium as gym
    import env  # this triggers registration
    environment = gym.make("HomingAgent-v0")
"""

from gymnasium.envs.registration import register

register(
    id="HomingAgent-v0",
    entry_point="env.homing_env:HomingEnv",
)
