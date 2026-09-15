"""Visual overlay and video annotation rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
import cv2
import numpy as np

from roadwatch.risk_engine import RiskAssessment
from roadwatch.tracks import TrackState
from roadwatch.types import RiskLevel
from roadwatch.zones import ZoneManager

if TYPE_CHECKING:
    from roadwatch.calibration import PerspectiveCalibrator

COLOR_PEDESTRIAN = (255, 178, 50)   # Blue-Cyan in BGR
COLOR_VEHICLE = (50, 150, 255)      # Orange in BGR
COLOR_TEXT = (255, 255, 255)
COLOR_HUD_BG = (25, 25, 25)

RISK_COLORS = {
    RiskLevel.LOW: (100, 200, 100),       # Green
    RiskLevel.MODERATE: (0, 215, 255),    # Yellow
    RiskLevel.HIGH: (0, 140, 255),        # Orange
    RiskLevel.CRITICAL: (0, 0, 255),      # Red
}


class Visualizer:
    """Renders bounding boxes, labels, trajectory trails, zones, and risk overlays."""

    def __init__(self, show_trails: bool = True, max_trail_points: int = 30) -> None:
        self.show_trails = show_trails
        self.max_trail_points = max_trail_points

    def draw_trail(self, frame: np.ndarray, track: TrackState) -> None:
        """Draw historical trajectory trail for a track."""
        if not self.show_trails or len(track.trajectory) < 2:
            return
        pts = [p.as_int_tuple() for p in track.trajectory[-self.max_trail_points:]]
        color = COLOR_PEDESTRIAN if track.is_pedestrian else COLOR_VEHICLE
        for i in range(1, len(pts)):
            thickness = max(1, int(2 * (i / len(pts))))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    def draw_track(
        self,
        frame: np.ndarray,
        track: TrackState,
        speed_kmh: Optional[float] = None,
    ) -> None:
        """Render bounding box, label badge, speed, and ground contact point."""
        x1, y1, x2, y2 = track.bounding_box.as_int_xyxy()
        color = COLOR_PEDESTRIAN if track.is_pedestrian else COLOR_VEHICLE

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        gx, gy = track.ground_point.as_int_tuple()
        cv2.circle(frame, (gx, gy), 4, color, -1)

        speed_tag = f" {speed_kmh:.0f}km/h" if speed_kmh is not None and speed_kmh > 1.0 else ""
        label = f"#{track.track_id} {track.class_name}{speed_tag}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, y1 - th - baseline - 4)
        badge_x2 = min(frame.shape[1], x1 + tw + 6)
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    def draw_risk_interaction(
        self,
        frame: np.ndarray,
        assessment: RiskAssessment,
        tracks_by_id: dict[int, TrackState],
    ) -> None:
        """Draw warning line and risk indicator badge between interacting pair."""
        if assessment.risk_level == RiskLevel.LOW:
            return

        ped = tracks_by_id.get(assessment.pedestrian_id)
        veh = tracks_by_id.get(assessment.vehicle_id)
        if not ped or not veh:
            return

        p1 = ped.ground_point.as_int_tuple()
        p2 = veh.ground_point.as_int_tuple()
        color = RISK_COLORS.get(assessment.risk_level, (0, 0, 255))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        dist_str = f" | {assessment.metric_distance:.1f}m" if assessment.metric_distance is not None else ""
        badge = f"{assessment.risk_level.value.upper()}{dist_str}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (mid_x - 4, mid_y - th - 4), (mid_x + tw + 4, mid_y + 4), color, -1)
        cv2.putText(frame, badge, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        fps: float,
        num_ped: int,
        num_veh: int,
        num_risks: int,
    ) -> None:
        """Render upper status HUD bar."""
        h, w = frame.shape[:2]
        hud_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        text = (
            f"Frame: {frame_idx:04d} | Time: {timestamp:05.1f}s | FPS: {fps:04.1f} | "
            f"Pedestrians: {num_ped} | Vehicles: {num_veh} | Active Risks: {num_risks}"
        )
        cv2.putText(frame, text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        tracks: List[TrackState],
        frame_idx: int,
        timestamp: float,
        fps: float,
        zone_mgr: Optional[ZoneManager] = None,
        assessments: Optional[List[RiskAssessment]] = None,
        speeds_by_id: Optional[Dict[int, float]] = None,
        calibrator: Optional[PerspectiveCalibrator] = None,
    ) -> np.ndarray:
        """Annotate frame with zones, tracks, trails, risk pairs, HUD, and optional BEV radar."""
        annotated = frame.copy()

        if zone_mgr:
            annotated = zone_mgr.draw_zones(annotated)

        tracks_by_id = {t.track_id: t for t in tracks}
        num_ped = sum(1 for t in tracks if t.is_pedestrian)
        num_veh = sum(1 for t in tracks if t.is_vehicle)

        for track in tracks:
            self.draw_trail(annotated, track)
            speed = speeds_by_id.get(track.track_id) if speeds_by_id else None
            self.draw_track(annotated, track, speed_kmh=speed)

        active_risks = 0
        if assessments:
            for a in assessments:
                if a.risk_level != RiskLevel.LOW:
                    active_risks += 1
                    self.draw_risk_interaction(annotated, a, tracks_by_id)

        self.draw_hud(annotated, frame_idx, timestamp, fps, num_ped, num_veh, active_risks)

        # Inset BEV radar canvas in top-right corner
        if calibrator:
            bev = calibrator.render_bev(tracks, bev_size=(150, 150))
            h, w = annotated.shape[:2]
            y1, y2 = 40, 190
            x1, x2 = w - 160, w - 10
            annotated[y1:y2, x1:x2] = bev

        return annotated
