"""Unit tests for zone definitions, querying, and membership."""

from roadwatch.config import SpatialConfig, ZoneItemConfig
from roadwatch.types import Point, ZoneType
from roadwatch.zones import ZoneManager


def test_zone_query_membership():
    """Verify ground point correctly detects zone membership and zone types."""
    cfg = SpatialConfig(
        zones={
            "sidewalk": ZoneItemConfig(
                type=ZoneType.PEDESTRIAN_ONLY,
                points=[(0, 0), (200, 0), (200, 100), (0, 100)],
            ),
            "crosswalk": ZoneItemConfig(
                type=ZoneType.SHARED_RISK,
                points=[(200, 0), (400, 0), (400, 100), (200, 100)],
            ),
        }
    )
    zm = ZoneManager(cfg)

    # Point in sidewalk
    p_side = Point(100, 50)
    zones = zm.get_zones_for_point(p_side)
    assert len(zones) == 1
    assert zones[0].name == "sidewalk"
    assert zm.is_in_zone_type(p_side, ZoneType.PEDESTRIAN_ONLY) is True
    assert zm.is_in_zone_type(p_side, ZoneType.SHARED_RISK) is False

    # Point in crosswalk
    p_cross = Point(300, 50)
    assert zm.is_in_zone_type(p_cross, ZoneType.SHARED_RISK) is True

    # Point outside all zones
    p_out = Point(500, 500)
    assert len(zm.get_zones_for_point(p_out)) == 0
