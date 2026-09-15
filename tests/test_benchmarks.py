"""Unit tests for benchmark dataset validation and scenario generators."""

import json
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_benchmark_scenarios import generate_crossing_conflict_video


def test_ground_truth_schema():
    """Verify that data/annotations/ground_truth_events.json conforms to schema."""
    gt_file = Path(__file__).resolve().parent.parent / "data" / "annotations" / "ground_truth_events.json"
    assert gt_file.exists()

    data = json.loads(gt_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1

    for item in data:
        assert "event_id" in item
        assert "start_time" in item
        assert "end_time" in item
        assert "zone" in item
        assert item["start_time"] <= item["end_time"]


def test_benchmark_scenario_generation(tmp_path):
    """Verify scenario generator writes readable MP4 video."""
    out_video = tmp_path / "test_scenario.mp4"
    generate_crossing_conflict_video(out_video, fps=10.0, duration_sec=1.0)

    assert out_video.exists()
    assert out_video.stat().st_size > 1000
