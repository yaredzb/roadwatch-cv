"""Unit tests for output exporters, evidence buffers, and summary reports."""

import json
import numpy as np
from roadwatch.events import EventRecord, EventState
from roadwatch.outputs import OutputExporter, RollingFrameBuffer
from roadwatch.types import RiskLevel


def test_rolling_frame_buffer():
    """Verify rolling buffer stores and limits frames correctly."""
    buf = RollingFrameBuffer(max_frames=3)
    f = np.zeros((10, 10, 3), dtype=np.uint8)

    buf.append(0.1, f)
    buf.append(0.2, f)
    assert len(buf.get_frames()) == 2

    buf.append(0.3, f)
    buf.append(0.4, f)
    frames = buf.get_frames()
    assert len(frames) == 3
    assert frames[0][0] == 0.2
    assert frames[-1][0] == 0.4


def test_output_exporter_json_and_csv(tmp_path):
    """Verify JSON and CSV event serialization."""
    exporter = OutputExporter(output_dir=tmp_path)

    events = [
        EventRecord(
            event_id="EVT-00001",
            pedestrian_id=1,
            vehicle_id=2,
            start_time=10.0,
            end_time=12.5,
            state=EventState.RESOLVED,
            highest_score=5,
            highest_risk_level=RiskLevel.HIGH,
            triggered_rules={"shared_zone_conflict", "close_proximity"},
            zone="crosswalk",
            min_normalized_distance=0.05,
        )
    ]

    # JSON test
    json_file = exporter.export_events_json(events)
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["event_id"] == "EVT-00001"
    assert data[0]["highest_risk_level"] == "high"
    assert data[0]["duration"] == 2.5

    # CSV test
    csv_file = exporter.export_events_csv(events)
    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # Header + 1 record
    assert "EVT-00001" in lines[1]
    assert "high" in lines[1]


def test_output_exporter_summary(tmp_path):
    """Verify summary metrics calculation and JSON output."""
    exporter = OutputExporter(output_dir=tmp_path)

    events = [
        EventRecord(
            event_id="EVT-00001",
            pedestrian_id=1,
            vehicle_id=2,
            start_time=1.0,
            end_time=3.0,
            highest_risk_level=RiskLevel.HIGH,
            triggered_rules={"close_proximity"},
            zone="zone_a",
        ),
        EventRecord(
            event_id="EVT-00002",
            pedestrian_id=3,
            vehicle_id=4,
            start_time=5.0,
            end_time=6.0,
            highest_risk_level=RiskLevel.CRITICAL,
            triggered_rules={"vehicle_in_pedestrian_zone"},
            zone="zone_b",
        ),
    ]

    summary_file = exporter.export_summary_json(
        events=events,
        total_frames=100,
        duration=5.0,
        avg_fps=20.0,
        total_pedestrians=2,
        total_vehicles=2,
        config_path="configs/default.yaml",
    )

    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["total_frames"] == 100
    assert summary["total_events"] == 2
    assert summary["events_by_severity"]["high"] == 1
    assert summary["events_by_severity"]["critical"] == 1
    assert summary["events_by_zone"]["zone_a"] == 1
