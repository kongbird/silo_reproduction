from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import torch
import yaml

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose

from silo.envs.harness_builder import build_u_harness


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@register_env(
    "SILOHarness-v1",
    max_episode_steps=32,
)
class SILOHarnessEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.sim_config = load_yaml("sim.yaml")
        self.harness_config = load_yaml("harness.yaml")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera = self.harness_config["camera"]

        pose = sapien_utils.look_at(
            eye=np.asarray(camera["eye"], dtype=np.float32),
            target=np.asarray(camera["target"], dtype=np.float32),
        )

        return [
            CameraConfig(
                uid="record_camera",
                pose=pose,
                width=int(camera["width"]),
                height=int(camera["height"]),
                fov=float(camera["fov"]),
                near=float(camera["near"]),
                far=float(camera["far"]),
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        return self._default_sensor_configs

    def _load_scene(self, options: dict):
        # ManiSkill table builder creates the work surface and background.
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        self.harness = build_u_harness(
            self.scene,
            self.harness_config,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            batch_size = len(env_idx)

            # Standard Panda arm configuration used only for scene inspection.
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0.0,
                    -5.0 * np.pi / 8,
                    0.0,
                    3.0 * np.pi / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ],
                dtype=np.float32,
            )

            qpos = torch.tensor(
                qpos,
                device=self.device,
            ).repeat(batch_size, 1)

            self.agent.reset(qpos)

            position = torch.tensor(
                self.harness_config["pose"]["position"],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            yaw = np.deg2rad(
                float(self.harness_config["pose"]["yaw_deg"])
            )

            quaternion = torch.tensor(
                [
                    np.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    np.sin(yaw / 2.0),
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            self.harness.set_pose(
                Pose.create_from_pq(
                    p=position,
                    q=quaternion,
                )
            )

    def evaluate(self):
        return {}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "harness_pose": self.harness.pose.raw_pose,
        }
