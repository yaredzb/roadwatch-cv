"""Core data models and type definitions for RoadWatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ZoneType(str, Enum):
    """Supported zone classifications."""
    PEDESTRIAN_ONLY = "pedestrian_only"
    VEHICLE_ZONE = "vehicle_zone"
    SHARED_RISK = "shared_risk"
    OBSERVATION = "observation"
    LINE_BOUNDARY = "line_boundary"


class RiskLevel(str, Enum):
    """Categorized risk levels based on cumulative risk score."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Point:
    """2D Cartesian point representation."""
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Ground contact point estimation (midpoint of bottom edge)."""
        return Point(x=(self.x1 + self.x2) / 2.0, y=self.y2)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass
class VideoMetadata:
    """Metadata extracted from video container."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # in seconds


@dataclass
class Detection:
    """Structured representation of a single object detector output."""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    ground_point: Point


@dataclass
class ZoneDefinition:
    """Spatial zone definition with ordered polygon coordinates."""
    name: str
    zone_type: ZoneType
    points: List[Point]
