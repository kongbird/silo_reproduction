from pathlib import Path

import numpy as np
import yaml


def main():
    with Path("configs/harness.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    position = np.asarray(
        config["pose"]["position"],
        dtype=np.float64,
    )

    radial_distance = np.linalg.norm(position[:2])
    distance_3d = np.linalg.norm(position)

    print("Harness position:", position)
    print("XY radial distance:", radial_distance)
    print("3D distance from robot base:", distance_3d)

    # Broad sanity limits, not an exact Panda IK test.
    assert 0.20 <= radial_distance <= 0.80, (
        "Harness is outside the broad Panda working region"
    )
    assert 0.0 <= position[2] <= 0.40, (
        "Harness height is unreasonable"
    )

    print("Broad reachability check passed.")
    print("An IK check is still required before motion execution.")


if __name__ == "__main__":
    main()
