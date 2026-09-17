from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import silo  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("videos/day2/harness_scene.png"),
    )
    return parser.parse_args()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "SILOHarness-v1",
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        sensor_configs={"record_camera": {}},
    )

    obs, info = env.reset(seed=0)

    print("Reset passed")
    print("Observation finite:", np.isfinite(to_numpy(obs)).all())
    print("Action shape:", env.action_space.shape)

    frame = env.render()
    frame = to_numpy(frame)

    # Vectorized rendering may include a leading environment dimension.
    if frame.ndim == 4:
        frame = frame[0]

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    imageio.imwrite(args.output, frame.astype(np.uint8))

    print("Saved:", args.output)
    print("Scene test passed")

    env.close()


if __name__ == "__main__":
    main()
