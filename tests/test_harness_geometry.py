import pytest
import yaml

from silo.envs.harness_builder import HarnessGeometry


def test_harness_dimensions():
    with open("configs/harness.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    geometry = HarnessGeometry.from_config(config)
    geometry.validate()

    assert geometry.inner_width == pytest.approx(0.024)
    assert geometry.pillar_height == pytest.approx(0.030)
    assert geometry.pillar_thickness == pytest.approx(0.006)
    assert geometry.depth == pytest.approx(0.018)
    assert geometry.bottom_thickness == pytest.approx(0.006)
    assert geometry.outer_width == pytest.approx(0.036)
