"""Geometric calculations, spatial distances, and motion trends."""

from __future__ import annotations

import math
from typing import List, Sequence
import cv2
import numpy as np

from roadwatch.types import Point


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Determine whether a 2D point lies strictly inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False
    pts = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
    # cv2.pointPolygonTest: >0 inside, ==0 on edge, <0 outside
    res = cv2.pointPolygonTest(pts, (point.x, point.y), measureDist=False)
    return res >= 0.0


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean pixel distance between two points."""
    return float(math.hypot(p2.x - p1.x, p2.y - p1.y))


def normalized_distance(p1: Point, p2: Point, frame_width: int, frame_height: int) -> float:
    """
    Calculate scale-invariant distance normalized by frame diagonal.
    Returns value in [0.0, 1.0].
    """
    diagonal = math.hypot(frame_width, frame_height)
    if diagonal <= 0.0:
        return 0.0
    return euclidean_distance(p1, p2) / diagonal


def is_closing_distance(distances: List[float], min_window: int = 3) -> bool:
    """
    Evaluate whether pairwise distance is consistently decreasing over recent sequence.
    Returns True if distance has monotonically decreased across at least min_window steps.
    """
    if len(distances) < min_window:
        return False

    recent = distances[-min_window:]
    decreases = 0
    for i in range(1, len(recent)):
        if recent[i] < recent[i - 1]:
            decreases += 1

    # Require at least 80% decreasing steps
    return (decreases / (len(recent) - 1)) >= 0.8
