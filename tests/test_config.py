"""Tests for configuration loading and validation."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from roadwatch.config import (
    AppConfig,
    load_config,
    load_config_from_dict,
)


def test_load_default_config():
    """Verify that configs/default.yaml loads and validates successfully."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = load_config(config_path)

    assert isinstance(cfg, AppConfig)
    assert cfg.video.input_path == "data/samples/sample.mp4"
    assert "person" in cfg.detection.relevant_classes
    assert cfg.tracking.algorithm == "bytetrack"
    assert len(cfg.spatial.zones) == 2
    assert "pedestrian_crossing" in cfg.spatial.zones
    assert cfg.events.min_persistence_sec == 0.5


def test_missing_config_file():
    """Verify that attempting to load a non-existent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_config.yaml")


def test_invalid_yaml_syntax(tmp_path):
    """Verify that corrupt YAML triggers a ValueError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("video:\n  input_path: [unclosed_bracket", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_config(bad_yaml)


def test_invalid_class_rejection():
    """Verify that unsupported class names raise a validation error."""
    raw = {
        "video": {"input_path": "input.mp4", "output_path": "outputs/"},
        "detection": {"relevant_classes": ["person", "spaceship"]},
    }
    with pytest.raises(ValidationError) as excinfo:
        load_config_from_dict(raw)
    assert "Unsupported class(es)" in str(excinfo.value)


def test_polygon_fewer_than_three_points():
    """Verify that polygons with fewer than 3 coordinates are rejected."""
    raw = {
        "video": {"input_path": "input.mp4", "output_path": "outputs/"},
        "spatial": {
            "zones": {
                "invalid_zone": {
                    "type": "shared_risk",
                    "points": [[10, 10], [20, 20]],
                }
            }
        },
    }
    with pytest.raises(ValidationError):
        load_config_from_dict(raw)


def test_negative_threshold_rejection():
    """Verify that negative thresholds raise validation errors."""
    raw = {
        "video": {"input_path": "input.mp4", "output_path": "outputs/"},
        "detection": {"confidence_threshold": -0.5},
    }
    with pytest.raises(ValidationError):
        load_config_from_dict(raw)


def test_contradictory_risk_thresholds():
    """Verify that non-monotonic risk thresholds are rejected."""
    raw = {
        "video": {"input_path": "input.mp4", "output_path": "outputs/"},
        "events": {
            "risk_thresholds": {
                "low": 5,
                "moderate": 2,  # moderate < low is contradictory
                "high": 4,
                "critical": 6,
            }
        },
    }
    with pytest.raises(ValidationError) as excinfo:
        load_config_from_dict(raw)
    assert "monotonically non-decreasing" in str(excinfo.value)
