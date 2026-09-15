"""Spatial zone management and polygon querying."""

from __future__ import annotations

from typing import Dict, List, Optional
import cv2
import numpy as np

from roadwatch.config import SpatialConfig
from roadwatch.geometry import point_in_polygon
from roadwatch.logger import get_logger
from roadwatch.types import Point, ZoneDefinition, ZoneType

logger = get_logger("roadwatch.zones")

ZONE_COLORS = {
    ZoneType.PEDESTRIAN_ONLY: (0, 0, 200),     # Red
    ZoneType.SHARED_RISK: (0, 200, 255),        # Yellow/Amber
    ZoneType.VEHICLE_ZONE: (180, 100, 50),     # Blue-ish
    ZoneType.OBSERVATION: (100, 100, 100),     # Gray
    ZoneType.LINE_BOUNDARY: (255, 0, 255),     # Magenta
}


class ZoneManager:
    """Manages scene zones, point-in-polygon tests, and zone visualizations."""

    def __init__(self, config: SpatialConfig) -> None:
        self.config = config
        self.zones: Dict[str, ZoneDefinition] = {}

        for name, item in config.zones.items():
            pts = [Point(x=p[0], y=p[1]) for p in item.points]
            self.zones[name] = ZoneDefinition(name=name, zone_type=item.type, points=pts)

        logger.info(f"Initialized {len(self.zones)} spatial zones: {list(self.zones.keys())}")

    def get_zones_for_point(self, point: Point) -> List[ZoneDefinition]:
        """Find all zones containing the given ground contact point."""
        return [z for z in self.zones.values() if point_in_polygon(point, z.points)]

    def is_in_zone_type(self, point: Point, zone_type: ZoneType) -> bool:
        """Check if a point lies in any zone of the specified type."""
        return any(z.zone_type == zone_type for z in self.get_zones_for_point(point))

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Render semi-transparent filled polygons and boundary lines onto the frame."""
        if not self.zones:
            return frame

        overlay = frame.copy()
        for zone in self.zones.values():
            pts_array = np.array([[int(round(p.x)), int(round(p.y))] for p in zone.points], dtype=np.int32)
            color = ZONE_COLORS.get(zone.zone_type, (128, 128, 128))

            # Fill zone
            cv2.fillPoly(overlay, [pts_array], color)
            # Outline
            cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=2)

            # Zone label at polygon centroid
            cx = int(np.mean([p.x for p in zone.points]))
            cy = int(np.mean([p.y for p in zone.points]))
            cv2.putText(
                frame, zone.name, (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
