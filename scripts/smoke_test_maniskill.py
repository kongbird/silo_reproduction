# scripts/smoke_test_maniskill.py
import gymnasium as gym
import torch
import mani_skill.envs

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

env = gym.make(
    "PickCube-v1",
    num_envs=16,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode=None,
)

obs, info = env.reset(seed=0)
print("Observation type:", type(obs))

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print("Smoke test passed")
env.close()
