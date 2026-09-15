"""Unit tests for spatial geometry, distances, and closing trends."""

import pytest
from roadwatch.geometry import (
    euclidean_distance,
    is_closing_distance,
    normalized_distance,
    point_in_polygon,
)
from roadwatch.types import Point


def test_point_in_polygon():
    """Verify point inside, outside, and on boundary of a polygon."""
    poly = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]

    # Strictly inside
    assert point_in_polygon(Point(50, 50), poly) is True

    # Strictly outside
    assert point_in_polygon(Point(150, 50), poly) is False
    assert point_in_polygon(Point(-10, 50), poly) is False

    # On boundary
    assert point_in_polygon(Point(0, 50), poly) is True
    assert point_in_polygon(Point(50, 0), poly) is True


def test_euclidean_and_normalized_distance():
    """Verify 2D Euclidean distance and frame-normalized scale-invariant distance."""
    p1 = Point(0, 0)
    p2 = Point(300, 400)

    # 3-4-5 right triangle -> dist = 500
    dist = euclidean_distance(p1, p2)
    assert dist == pytest.approx(500.0)

    # Frame 600x800 -> diagonal = 1000 -> norm = 500 / 1000 = 0.5
    norm = normalized_distance(p1, p2, frame_width=600, frame_height=800)
    assert norm == pytest.approx(0.5)


def test_is_closing_distance():
    """Verify detection of monotonic/consistent decrease in pairwise distance."""
    # Monotonically decreasing
    assert is_closing_distance([100.0, 90.0, 80.0, 70.0], min_window=3) is True

    # Increasing
    assert is_closing_distance([70.0, 80.0, 90.0, 100.0], min_window=3) is False

    # Fluctuating
    assert is_closing_distance([100.0, 105.0, 95.0, 102.0], min_window=3) is False

    # Insufficient history
    assert is_closing_distance([100.0, 90.0], min_window=3) is False
