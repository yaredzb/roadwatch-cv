"""Unit tests for trajectory management, smoothing, and track lifecycles."""

import pytest
from roadwatch.config import TrackingConfig
from roadwatch.tracks import TrajectoryManager
from roadwatch.types import BoundingBox, Point


def test_trajectory_smoothing():
    """Verify exponential moving average formula: alpha * current + (1 - alpha) * prev."""
    cfg = TrackingConfig(max_trajectory_length=10)
    tm = TrajectoryManager(config=cfg, smoothing_alpha=0.5)

    # First observation at (100, 100)
    b1 = BoundingBox(x1=90, y1=80, x2=110, y2=100)
    t1 = tm.update_track(track_id=1, class_name="person", confidence=0.9, bounding_box=b1, timestamp=0.0)
    assert t1.smoothed_point == Point(100.0, 100.0)

    # Second observation at (200, 200) -> 0.5 * 200 + 0.5 * 100 = 150
    b2 = BoundingBox(x1=190, y1=180, x2=210, y2=200)
    t2 = tm.update_track(track_id=1, class_name="person", confidence=0.9, bounding_box=b2, timestamp=0.1)
    assert t2.smoothed_point.x == pytest.approx(150.0)
    assert t2.smoothed_point.y == pytest.approx(150.0)


def test_trajectory_max_length_limit():
    """Verify that stored trajectory trail does not exceed configured max length."""
    cfg = TrackingConfig(max_trajectory_length=5)
    tm = TrajectoryManager(config=cfg, smoothing_alpha=0.3)

    for i in range(10):
        box = BoundingBox(x1=i * 10, y1=i * 10, x2=(i + 1) * 10, y2=(i + 1) * 10)
        track = tm.update_track(track_id=2, class_name="car", confidence=0.85, bounding_box=box, timestamp=float(i))

    assert len(track.trajectory) == 5


def test_track_expiration_on_missed_frames():
    """Verify tracks are purged after exceeding lost track buffer."""
    cfg = TrackingConfig(track_buffer=2, min_observations=1)
    tm = TrajectoryManager(config=cfg)

    box = BoundingBox(x1=10, y1=10, x2=20, y2=20)
    tm.update_track(track_id=3, class_name="person", confidence=0.9, bounding_box=box, timestamp=0.0)

    assert 3 in tm.active_tracks

    # Frame 1: missed
    tm.mark_missed(visible_track_ids=set())
    assert 3 in tm.active_tracks
    assert tm.active_tracks[3].missed_frames == 1

    # Frame 2: missed
    tm.mark_missed(visible_track_ids=set())
    assert 3 in tm.active_tracks
    assert tm.active_tracks[3].missed_frames == 2

    # Frame 3: exceeds buffer (2) -> expired and purged
    tm.mark_missed(visible_track_ids=set())
    assert 3 not in tm.active_tracks


def test_displacement_calculation():
    """Verify track displacement calculates Euclidean distance correctly."""
    cfg = TrackingConfig(max_trajectory_length=10)
    tm = TrajectoryManager(config=cfg, smoothing_alpha=1.0)  # alpha=1 means no smoothing lag

    box1 = BoundingBox(x1=0, y1=0, x2=10, y2=0)   # bottom_center = (5, 0)
    box2 = BoundingBox(x1=30, y1=40, x2=40, y2=40) # bottom_center = (35, 40) -> delta=(30, 40) -> dist=50

    t = tm.update_track(track_id=4, class_name="car", confidence=0.9, bounding_box=box1, timestamp=0.0)
    t = tm.update_track(track_id=4, class_name="car", confidence=0.9, bounding_box=box2, timestamp=0.1)

    assert t.displacement() == pytest.approx(50.0)
