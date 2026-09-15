"""Unit tests for perspective calibration, metric distance, and BEV generation."""

import pytest
from roadwatch.calibration import PerspectiveCalibrator
from roadwatch.tracks import TrackState
from roadwatch.types import BoundingBox, Point


def test_homography_projection():
    """Verify homography transformation maps pixel coordinates to accurate ground-plane meters."""
    # Source: 100x200 pixel rectangle
    src_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)]
    # Target: 10m x 20m physical rectangle
    calib = PerspectiveCalibrator(src_pts, target_dimensions_meters=(10.0, 20.0))

    # Corner 1: (0, 0) -> (0, 0)m
    g1 = calib.project_to_ground(Point(0.0, 0.0))
    assert g1.x == pytest.approx(0.0, abs=0.01)
    assert g1.y == pytest.approx(0.0, abs=0.01)

    # Corner 3: (100, 200) -> (10, 20)m
    g3 = calib.project_to_ground(Point(100.0, 200.0))
    assert g3.x == pytest.approx(10.0, abs=0.01)
    assert g3.y == pytest.approx(20.0, abs=0.01)

    # Midpoint: (50, 100) -> (5, 10)m
    g_mid = calib.project_to_ground(Point(50.0, 100.0))
    assert g_mid.x == pytest.approx(5.0, abs=0.01)
    assert g_mid.y == pytest.approx(10.0, abs=0.01)


def test_metric_distance():
    """Verify physical distance calculation between two ground points."""
    src_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calib = PerspectiveCalibrator(src_pts, target_dimensions_meters=(10.0, 10.0))

    # 3-4-5 triangle in ground meters: (0, 0)m to (3, 4)m -> dist = 5.0m
    p1 = Point(0.0, 0.0)      # (0, 0)m
    p2 = Point(30.0, 40.0)    # (3, 4)m
    dist_m = calib.metric_distance(p1, p2)
    assert dist_m == pytest.approx(5.0, abs=0.05)


def test_speed_calculation():
    """Verify speed calculation converts displacement over time to km/h."""
    src_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calib = PerspectiveCalibrator(src_pts, target_dimensions_meters=(10.0, 10.0))

    # Object moves from (0, 0) to (100, 0) in 1.0 second -> 10 meters in 1 sec = 10 m/s = 36 km/h
    trajectory = [Point(0.0, 0.0), Point(50.0, 0.0), Point(100.0, 0.0)]
    timestamps = [0.0, 0.5, 1.0]

    speed = calib.calculate_speed_kmh(trajectory, timestamps, window=3)
    assert speed == pytest.approx(36.0, abs=0.5)


def test_bev_radar_rendering():
    """Verify Bird's-Eye View canvas generates proper image dimensions."""
    src_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calib = PerspectiveCalibrator(src_pts, target_dimensions_meters=(10.0, 10.0))

    track = TrackState(
        track_id=1,
        class_name="car",
        confidence=0.9,
        bounding_box=BoundingBox(10, 10, 30, 30),
        ground_point=Point(20, 30),
        smoothed_point=Point(20, 30),
    )

    bev = calib.render_bev([track], bev_size=(160, 160))
    assert bev.shape == (160, 160, 3)
