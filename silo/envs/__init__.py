from gymnasium.envs.registration import register
from silo.envs.cable_routing_env import SILOHarnessEnv

register(
    id="SILOHarness-v1",
    entry_point="silo.envs.cable_routing_env:SILOHarnessEnv",
    max_episode_steps=32,
)

__all__ = ["SILOHarnessEnv"]
