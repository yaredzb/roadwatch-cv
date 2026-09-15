"""Configuration schema and validation for RoadWatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from roadwatch.types import ZoneType

SUPPORTED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck"
}


class VideoConfig(BaseModel):
    """Configuration for video I/O."""
    input_path: str = Field(..., description="Path to input video file")
    output_path: str = Field(..., description="Directory or path for output artifacts")
    resize_width: Optional[int] = Field(default=None, gt=0)
    resize_height: Optional[int] = Field(default=None, gt=0)
    frame_skip: int = Field(default=0, ge=0)
    output_codec: str = Field(default="mp4v", min_length=4, max_length=4)
    live_preview: bool = Field(default=False)


class DetectionConfig(BaseModel):
    """Configuration for YOLO detector."""
    model_path: str = Field(default="yolov8n.pt")
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    relevant_classes: List[str] = Field(
        default_factory=lambda: ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    )
    device: str = Field(default="cpu")
    img_size: int = Field(default=640, gt=0)

    @field_validator("relevant_classes")
    @classmethod
    def validate_classes(cls, classes: List[str]) -> List[str]:
        if not classes:
            raise ValueError("At least one target class must be specified.")
        unknown = [c for c in classes if c.lower() not in SUPPORTED_CLASSES]
        if unknown:
            raise ValueError(
                f"Unsupported class(es): {unknown}. Supported classes: {sorted(SUPPORTED_CLASSES)}"
            )
        return [c.lower() for c in classes]


class TrackingConfig(BaseModel):
    """Configuration for ByteTrack tracker."""
    algorithm: str = Field(default="bytetrack")
    track_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, gt=0)
    min_observations: int = Field(default=3, ge=1)
    max_trajectory_length: int = Field(default=60, gt=1)


class ZoneItemConfig(BaseModel):
    """Definition of an individual polygon zone."""
    type: ZoneType
    points: List[Tuple[float, float]] = Field(..., min_length=3)

    @field_validator("points")
    @classmethod
    def validate_points(cls, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pts) < 3:
            raise ValueError("Polygon must contain at least 3 coordinate points.")
        return pts


class SpatialConfig(BaseModel):
    """Configuration for zones and proximity analysis."""
    zones: Dict[str, ZoneItemConfig] = Field(default_factory=dict)
    distance_threshold: float = Field(default=0.08, gt=0.0)
    trajectory_smoothing_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    movement_threshold_px: float = Field(default=3.0, ge=0.0)
    closing_distance_window: int = Field(default=5, ge=2)


class EventConfig(BaseModel):
    """Configuration for risk rules and event lifecycle."""
    min_persistence_sec: float = Field(default=0.5, ge=0.0)
    cooldown_sec: float = Field(default=3.0, ge=0.0)
    pre_event_clip_sec: float = Field(default=2.0, ge=0.0)
    post_event_clip_sec: float = Field(default=2.0, ge=0.0)
    snapshot_generation: bool = Field(default=True)
    enabled_rules: List[str] = Field(
        default_factory=lambda: [
            "shared_zone_conflict",
            "vehicle_in_pedestrian_zone",
            "close_proximity",
            "closing_distance",
            "sustained_exposure",
        ]
    )
    risk_weights: Dict[str, int] = Field(
        default_factory=lambda: {
            "shared_zone_conflict": 1,
            "vehicle_in_pedestrian_zone": 3,
            "close_proximity": 2,
            "closing_distance": 2,
            "vehicle_moving": 1,
            "sustained_exposure": 1,
        }
    )
    risk_thresholds: Dict[str, int] = Field(
        default_factory=lambda: {
            "low": 0,
            "moderate": 2,
            "high": 4,
            "critical": 6,
        }
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> EventConfig:
        t = self.risk_thresholds
        if not (t.get("low", 0) <= t.get("moderate", 2) <= t.get("high", 4) <= t.get("critical", 6)):
            raise ValueError("Risk score thresholds must be strictly monotonically non-decreasing.")
        return self


class CalibrationConfig(BaseModel):
    """Configuration for planar homography calibration."""
    enabled: bool = Field(default=False)
    source_points: Optional[List[Tuple[float, float]]] = Field(default=None)
    target_dimensions_meters: Optional[Tuple[float, float]] = Field(default=None)
    metric_distance_threshold: float = Field(default=2.5, gt=0.0)
    speed_threshold_kmh: float = Field(default=20.0, gt=0.0)

    @field_validator("source_points")
    @classmethod
    def validate_points(cls, pts: Optional[List[Tuple[float, float]]]) -> Optional[List[Tuple[float, float]]]:
        if pts is not None and len(pts) != 4:
            raise ValueError("Calibration requires exactly 4 reference points.")
        return pts


class AppConfig(BaseModel):
    """Complete application configuration model."""
    video: VideoConfig
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


def load_config_from_dict(raw: Dict[str, Any]) -> AppConfig:
    """Instantiate and validate AppConfig from a dictionary."""
    return AppConfig(**raw)


def load_config(path: str | Path) -> AppConfig:
    """Load, parse, and validate application configuration from YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file root must be a dictionary, got {type(data).__name__}")

    return load_config_from_dict(data)
