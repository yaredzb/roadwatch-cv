"""Integration tests for end-to-end pipeline execution and evaluation metrics."""

import json
import cv2
import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_events import compute_metrics, match_events
from roadwatch.config import load_config
from scripts.analyze_video import process_video_stream


def test_event_matching_and_metrics():
    """Verify precision, recall, F1, and duplicate calculation logic."""
    gt = [
        {"event_id": "GT-1", "start_time": 10.0, "end_time": 15.0, "zone": "crossing"},
        {"event_id": "GT-2", "start_time": 30.0, "end_time": 35.0, "zone": "sidewalk"},
    ]

    # Predictions: 1 TP, 1 Duplicate of GT-1, 1 FP
    preds = [
        {"event_id": "EVT-1", "start_time": 11.0, "end_time": 14.0, "zone": "crossing"},
        {"event_id": "EVT-2", "start_time": 12.0, "end_time": 13.0, "zone": "crossing"}, # Duplicate
        {"event_id": "EVT-3", "start_time": 50.0, "end_time": 52.0, "zone": "crossing"}, # FP
    ]

    tp, fp, fn, dups = match_events(preds, gt, time_tolerance=2.0)
    assert tp == 1
    assert dups == 1
    assert fn == 1
    assert fp == 2 # 1 FP + 1 Duplicate counted as false alert

    metrics = compute_metrics(tp, fp, fn, dups, duration_sec=60.0)
    assert metrics["precision"] == pytest.approx(1 / 3, abs=0.01)
    assert metrics["recall"] == pytest.approx(0.5, abs=0.01)


def test_full_pipeline_integration(tmp_path):
    """Verify end-to-end pipeline produces video, JSON, CSV, and summary artifacts."""
    test_video = tmp_path / "integration_sample.mp4"
    width, height, fps, frames = 320, 240, 10.0, 15
    out = cv2.VideoWriter(str(test_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        out.write(np.full((height, width, 3), (i * 15 % 255, 100, 200), dtype=np.uint8))
    out.release()

    default_config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = load_config(default_config_path)

    out_dir = tmp_path / "integration_output"
    out_dir.mkdir()

    import logging
    logger = logging.getLogger("test_integration")

    process_video_stream(
        cfg=cfg,
        in_file=test_video,
        out_dir=out_dir,
        max_frames=15,
        disable_clips=True,
        logger=logger,
    )

    assert (out_dir / f"annotated_{test_video.name}").exists()
    assert (out_dir / "events.json").exists()
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "summary.json").exists()

    summary_data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_data["total_frames"] == 15
