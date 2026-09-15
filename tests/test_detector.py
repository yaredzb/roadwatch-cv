"""Unit tests for YOLO detector wrapper and bounding box geometry."""

from roadwatch.config import DetectionConfig
from roadwatch.detector import YOLODetector
from roadwatch.types import BoundingBox, Point


def test_bounding_box_properties():
    """Verify BoundingBox geometry, width, height, center, and bottom_center."""
    box = BoundingBox(x1=100.0, y1=200.0, x2=200.0, y2=400.0)
    assert box.width == 100.0
    assert box.height == 200.0
    assert box.center == Point(150.0, 300.0)
    assert box.bottom_center == Point(150.0, 400.0)


def test_bounding_box_clipping():
    """Verify that detector box clipping constrains coordinates to frame boundaries."""
    cfg = DetectionConfig()
    # Mock-free clipping test directly via internal method
    detector = YOLODetector.__new__(YOLODetector)

    # Box exceeding bottom-right boundary
    clipped = detector._clip_box((500.0, 400.0, 700.0, 550.0), width=640, height=480)
    assert clipped.x1 == 500.0
    assert clipped.y1 == 400.0
    assert clipped.x2 == 640.0
    assert clipped.y2 == 480.0

    # Negative coordinates
    clipped_neg = detector._clip_box((-50.0, -20.0, 100.0, 150.0), width=640, height=480)
    assert clipped_neg.x1 == 0.0
    assert clipped_neg.y1 == 0.0
    assert clipped_neg.x2 == 100.0
    assert clipped_neg.y2 == 150.0
