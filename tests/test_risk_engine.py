"""Unit tests for rule-based risk evaluation."""

from roadwatch.config import EventConfig, SpatialConfig, ZoneItemConfig
from roadwatch.risk_engine import RiskEngine
from roadwatch.tracks import TrackState
from roadwatch.types import BoundingBox, Point, RiskLevel, ZoneType
from roadwatch.zones import ZoneManager


def test_risk_rule_evaluation():
    """Verify individual rules, score summation, and risk classification."""
    spatial_cfg = SpatialConfig(
        distance_threshold=0.1,
        movement_threshold_px=5.0,
        closing_distance_window=3,
        zones={
            "ped_zone": ZoneItemConfig(
                type=ZoneType.PEDESTRIAN_ONLY,
                points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ),
            "shared": ZoneItemConfig(
                type=ZoneType.SHARED_RISK,
                points=[(100, 0), (300, 0), (300, 100), (100, 100)],
            ),
        },
    )
    event_cfg = EventConfig()
    engine = RiskEngine(event_cfg, spatial_cfg)
    zm = ZoneManager(spatial_cfg)

    # 1. Pedestrian and moving vehicle in shared zone
    ped = TrackState(
        track_id=1,
        class_name="person",
        confidence=0.9,
        bounding_box=BoundingBox(120, 40, 140, 60),
        ground_point=Point(130, 60),
        smoothed_point=Point(130, 60),
    )
    veh = TrackState(
        track_id=2,
        class_name="car",
        confidence=0.9,
        bounding_box=BoundingBox(150, 40, 190, 60),
        ground_point=Point(170, 60),
        smoothed_point=Point(170, 60),
        trajectory=[Point(160, 60), Point(170, 60)],  # movement = 10px
    )

    assessment = engine.evaluate_pair(
        ped=ped,
        veh=veh,
        distance_history=[50.0, 45.0, 40.0],
        zone_mgr=zm,
        frame_width=640,
        frame_height=480,
    )

    # Should trigger shared_zone_conflict (+1) + close_proximity (+2) + closing_distance (+2) = 5 (HIGH)
    assert "shared_zone_conflict" in assessment.triggered_rules
    assert "close_proximity" in assessment.triggered_rules
    assert "closing_distance" in assessment.triggered_rules
    assert assessment.score >= 5
    assert assessment.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
