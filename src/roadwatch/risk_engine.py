"""Rule-based risk assessment engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional
from roadwatch.config import EventConfig, SpatialConfig
from roadwatch.geometry import is_closing_distance, normalized_distance
from roadwatch.logger import get_logger
from roadwatch.tracks import TrackState
from roadwatch.types import RiskLevel, ZoneType
from roadwatch.zones import ZoneManager

if TYPE_CHECKING:
    from roadwatch.calibration import PerspectiveCalibrator

logger = get_logger("roadwatch.risk")


@dataclass
class RiskAssessment:
    """Outcome of risk rule evaluation for a pedestrian-vehicle pair."""
    pedestrian_id: int
    vehicle_id: int
    score: int
    risk_level: RiskLevel
    triggered_rules: List[str]
    normalized_distance: float
    shared_zones: List[str]
    metric_distance: Optional[float] = None
    vehicle_speed_kmh: Optional[float] = None


class RiskEngine:
    """Evaluates spatial rules and calculates cumulative risk scores."""

    def __init__(self, event_config: EventConfig, spatial_config: SpatialConfig) -> None:
        self.event_cfg = event_config
        self.spatial_cfg = spatial_config
        self.weights = event_config.risk_weights
        self.thresholds = event_config.risk_thresholds
        self.enabled_rules = set(event_config.enabled_rules)

    def _classify_level(self, score: int) -> RiskLevel:
        """Map cumulative integer score to categorized RiskLevel."""
        if score >= self.thresholds.get("critical", 6):
            return RiskLevel.CRITICAL
        if score >= self.thresholds.get("high", 4):
            return RiskLevel.HIGH
        if score >= self.thresholds.get("moderate", 2):
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def evaluate_pair(
        self,
        ped: TrackState,
        veh: TrackState,
        distance_history: List[float],
        zone_mgr: ZoneManager,
        frame_width: int,
        frame_height: int,
        interaction_duration: float = 0.0,
        calibrator: Optional[PerspectiveCalibrator] = None,
        metric_threshold: float = 2.5,
        veh_timestamps: Optional[List[float]] = None,
    ) -> RiskAssessment:
        """Evaluate all enabled risk rules for a specific pedestrian-vehicle pair."""
        score = 0
        triggered: List[str] = []

        norm_dist = normalized_distance(
            ped.smoothed_point, veh.smoothed_point, frame_width, frame_height
        )

        metric_dist = None
        speed_kmh = None
        if calibrator:
            metric_dist = calibrator.metric_distance(ped.smoothed_point, veh.smoothed_point)
            if veh_timestamps and len(veh_timestamps) >= 2:
                speed_kmh = calibrator.calculate_speed_kmh(veh.trajectory, veh_timestamps)

        ped_zones = set(z.name for z in zone_mgr.get_zones_for_point(ped.smoothed_point))
        veh_zones = set(z.name for z in zone_mgr.get_zones_for_point(veh.smoothed_point))
        common_zones = list(ped_zones.intersection(veh_zones))

        veh_moving = veh.displacement() >= self.spatial_cfg.movement_threshold_px

        # Rule 1: Shared-zone conflict
        if "shared_zone_conflict" in self.enabled_rules and common_zones:
            has_shared_risk = any(
                zone_mgr.zones[z_name].zone_type == ZoneType.SHARED_RISK
                for z_name in common_zones
            )
            if has_shared_risk and veh_moving:
                score += self.weights.get("shared_zone_conflict", 1)
                triggered.append("shared_zone_conflict")

        # Rule 2: Vehicle in pedestrian zone
        if "vehicle_in_pedestrian_zone" in self.enabled_rules:
            if zone_mgr.is_in_zone_type(veh.smoothed_point, ZoneType.PEDESTRIAN_ONLY):
                score += self.weights.get("vehicle_in_pedestrian_zone", 3)
                triggered.append("vehicle_in_pedestrian_zone")

        # Rule 3: Close proximity (metric if calibrated, normalized otherwise)
        if "close_proximity" in self.enabled_rules:
            is_close = (metric_dist < metric_threshold) if metric_dist is not None else (norm_dist < self.spatial_cfg.distance_threshold)
            if is_close:
                score += self.weights.get("close_proximity", 2)
                triggered.append("close_proximity")

        # Rule 4: Closing-distance interaction
        if "closing_distance" in self.enabled_rules and veh_moving:
            if is_closing_distance(distance_history, min_window=self.spatial_cfg.closing_distance_window):
                score += self.weights.get("closing_distance", 2)
                triggered.append("closing_distance")

        # Rule 5: Sustained exposure
        if "sustained_exposure" in self.enabled_rules:
            if interaction_duration >= 2.0 and (triggered or score > 0):
                score += self.weights.get("sustained_exposure", 1)
                triggered.append("sustained_exposure")

        risk_level = self._classify_level(score)

        return RiskAssessment(
            pedestrian_id=ped.track_id,
            vehicle_id=veh.track_id,
            score=score,
            risk_level=risk_level,
            triggered_rules=triggered,
            normalized_distance=norm_dist,
            shared_zones=common_zones,
            metric_distance=metric_dist,
            vehicle_speed_kmh=speed_kmh,
        )
