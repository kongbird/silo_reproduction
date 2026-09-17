from __future__ import annotations

from dataclasses import dataclass
from math import radians,cos,sin
from typing import Any

import sapien
from sapien.physx import PhysxMaterial
import torch

from mani_skill.utils.structs import Pose


@dataclass(frozen=True)
class HarnessGeometry:
    inner_width: float
    pillar_height: float
    pillar_thickness: float
    depth: float
    bottom_thickness: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HarnessGeometry":
        geometry = config["geometry"]
        return cls(
            inner_width=float(geometry["inner_width"]),
            pillar_height=float(geometry["pillar_height"]),
            pillar_thickness=float(geometry["pillar_thickness"]),
            depth=float(geometry["depth"]),
            bottom_thickness=float(geometry["bottom_thickness"]),
        )

    @property
    def outer_width(self) -> float:
        return self.inner_width + 2.0 * self.pillar_thickness

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


def _add_box(
    builder,
    *,
    half_size: list[float],
    local_position: list[float],
    color: list[float],
    material,
) -> None:
    local_pose = sapien.Pose(p=local_position)

    builder.add_box_collision(
        pose=local_pose,
        half_size=half_size,
        material=material,
    )
    builder.add_box_visual(
        pose=local_pose,
        half_size=half_size,
        material=color,
    )


def build_u_harness(scene, config: dict[str, Any]):
    geometry = HarnessGeometry.from_config(config)
    geometry.validate()

    appearance = config["appearance"]
    color = [float(v) for v in appearance["color_rgba"]]

    material_config = config["material"]
    physical_material = PhysxMaterial(
    static_friction=float(material_config["static_friction"]),
    dynamic_friction=float(material_config["dynamic_friction"]),
    restitution=float(material_config["restitution"]),
)

    builder = scene.create_actor_builder()

    # Harness local origin is at the bottom face of the bottom bar.
    bottom_half_size = [
        geometry.outer_width / 2.0,
        geometry.depth / 2.0,
        geometry.bottom_thickness / 2.0,
    ]
    bottom_position = [
        0.0,
        0.0,
        geometry.bottom_thickness / 2.0,
    ]

    pillar_half_size = [
        geometry.pillar_thickness / 2.0,
        geometry.depth / 2.0,
        geometry.pillar_height / 2.0,
    ]

    pillar_center_z = (
        geometry.bottom_thickness
        + geometry.pillar_height / 2.0
    )
    pillar_center_x = (
        geometry.inner_width / 2.0
        + geometry.pillar_thickness / 2.0
    )

    _add_box(
        builder,
        half_size=bottom_half_size,
        local_position=bottom_position,
        color=color,
        material=physical_material,
    )
    _add_box(
        builder,
        half_size=pillar_half_size,
        local_position=[-pillar_center_x, 0.0, pillar_center_z],
        color=color,
        material=physical_material,
    )
    _add_box(
        builder,
        half_size=pillar_half_size,
        local_position=[pillar_center_x, 0.0, pillar_center_z],
        color=color,
        material=physical_material,
    )

    pose_config = config["pose"]
    position = [float(v) for v in pose_config["position"]]
    yaw = radians(float(pose_config["yaw_deg"]))

    builder.initial_pose = sapien.Pose(
        p=position,
        # SAPIEN quaternion order: [w, x, y, z]
        q=[
            cos(yaw / 2.0),
            0.0,
            0.0,
            sin(yaw / 2.0),
        ],
    )

    return builder.build_kinematic(name="u_harness")
