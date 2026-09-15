"""Perspective calibration, homography transformation, and Bird's-Eye View (BEV)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from roadwatch.logger import get_logger
from roadwatch.tracks import TrackState
from roadwatch.types import Point

logger = get_logger("roadwatch.calibration")


class PerspectiveCalibrator:
    """Computes homography transformation mapping camera pixels to ground-plane meters."""

    def __init__(
        self,
        source_points: List[Tuple[float, float]],
        target_dimensions_meters: Tuple[float, float],
    ) -> None:
        if len(source_points) != 4:
            raise ValueError("Perspective calibration requires exactly 4 reference points.")

        self.source_points = source_points
        self.width_meters, self.length_meters = target_dimensions_meters

        src_pts = np.array(source_points, dtype=np.float32)
        dst_pts = np.array([
            [0.0, 0.0],
            [self.width_meters, 0.0],
            [self.width_meters, self.length_meters],
            [0.0, self.length_meters],
        ], dtype=np.float32)

        self.homography_matrix, status = cv2.findHomography(src_pts, dst_pts)
        if self.homography_matrix is None:
            raise ValueError("Failed to compute valid homography matrix from points.")

        logger.info(
            f"Perspective calibration initialized: {self.width_meters}m x {self.length_meters}m plane"
        )

    def project_to_ground(self, point: Point) -> Point:
        """Transform a 2D camera pixel point to metric ground plane coordinates (meters)."""
        pt_homo = np.array([[[point.x, point.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_homo, self.homography_matrix)
        gx, gy = transformed[0][0]
        return Point(x=float(gx), y=float(gy))

    def metric_distance(self, p1: Point, p2: Point) -> float:
        """Compute physical Euclidean distance in meters between two camera ground points."""
        g1 = self.project_to_ground(p1)
        g2 = self.project_to_ground(p2)
        return float(math.hypot(g2.x - g1.x, g2.y - g1.y))

    def calculate_speed_kmh(
        self,
        trajectory: List[Point],
        timestamps: List[float],
        window: int = 5,
    ) -> float:
        """
        Estimate object ground speed (km/h) across recent trajectory observations.
        v = (delta_distance_meters / delta_time_seconds) * 3.6
        """
        if len(trajectory) < 2 or len(timestamps) < 2:
            return 0.0

        n = min(window, len(trajectory), len(timestamps))
        dt = timestamps[-1] - timestamps[-n]
        if dt <= 1e-4:
            return 0.0

        p_start_m = self.project_to_ground(trajectory[-n])
        p_end_m = self.project_to_ground(trajectory[-1])
        dist_meters = math.hypot(p_end_m.x - p_start_m.x, p_end_m.y - p_start_m.y)

        speed_mps = dist_meters / dt
        return float(speed_mps * 3.6)

    def render_bev(
        self,
        tracks: List[TrackState],
        bev_size: Tuple[int, int] = (160, 160),
    ) -> np.ndarray:
        """Render a top-down Bird's-Eye View (BEV) radar canvas."""
        canvas = np.full((bev_size[1], bev_size[0], 3), (30, 30, 30), dtype=np.uint8)

        # Draw border
        cv2.rectangle(canvas, (0, 0), (bev_size[0] - 1, bev_size[1] - 1), (80, 80, 80), 1)
        cv2.putText(canvas, "BEV Radar", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        scale_x = (bev_size[0] - 20) / max(1.0, self.width_meters)
        scale_y = (bev_size[1] - 30) / max(1.0, self.length_meters)

        for track in tracks:
            g_pt = self.project_to_ground(track.smoothed_point)
            bx = int(round(10 + g_pt.x * scale_x))
            by = int(round(20 + g_pt.y * scale_y))

            # Clamp to canvas
            bx = max(4, min(bev_size[0] - 4, bx))
            by = max(18, min(bev_size[1] - 4, by))

            color = (255, 178, 50) if track.is_pedestrian else (50, 150, 255)
            cv2.circle(canvas, (bx, by), 4, color, -1)

        return canvas
