"""Trajectory and track state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from roadwatch.config import TrackingConfig
from roadwatch.logger import get_logger
from roadwatch.types import BoundingBox, Point

logger = get_logger("roadwatch.tracks")


@dataclass
class TrackState:
    """Maintains state, trajectory history, and smoothing for a single tracked object."""
    track_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    ground_point: Point
    smoothed_point: Point
    trajectory: List[Point] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    visible_frames: int = 1
    missed_frames: int = 0

    @property
    def is_pedestrian(self) -> bool:
        return self.class_name in {"person"}

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}

    def displacement(self, window: int = 5) -> float:
        """Calculate total Euclidean displacement across recent trajectory points."""
        if len(self.trajectory) < 2:
            return 0.0
        recent = self.trajectory[-min(window, len(self.trajectory)):]
        p_start = recent[0]
        p_end = recent[-1]
        return float(np.hypot(p_end.x - p_start.x, p_end.y - p_start.y))


class TrajectoryManager:
    """Manages active tracks, applies exponential smoothing, and purges expired tracks."""

    def __init__(self, config: TrackingConfig, smoothing_alpha: float = 0.3) -> None:
        self.config = config
        self.smoothing_alpha = smoothing_alpha
        self.active_tracks: Dict[int, TrackState] = {}

    def _smooth(self, current: Point, prev_smoothed: Point) -> Point:
        """Apply exponential moving average: alpha * current + (1 - alpha) * prev."""
        sx = self.smoothing_alpha * current.x + (1.0 - self.smoothing_alpha) * prev_smoothed.x
        sy = self.smoothing_alpha * current.y + (1.0 - self.smoothing_alpha) * prev_smoothed.y
        return Point(x=sx, y=sy)

    def update_track(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        bounding_box: BoundingBox,
        timestamp: float,
    ) -> TrackState:
        """Update or register a track with new detection observations."""
        ground_pt = bounding_box.bottom_center

        if track_id not in self.active_tracks:
            track = TrackState(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                bounding_box=bounding_box,
                ground_point=ground_pt,
                smoothed_point=ground_pt,
                trajectory=[ground_pt],
                first_seen=timestamp,
                last_seen=timestamp,
                visible_frames=1,
                missed_frames=0,
            )
            self.active_tracks[track_id] = track
            return track

        track = self.active_tracks[track_id]
        track.confidence = confidence
        track.bounding_box = bounding_box
        track.ground_point = ground_pt
        track.smoothed_point = self._smooth(ground_pt, track.smoothed_point)
        track.trajectory.append(track.smoothed_point)

        # Enforce maximum stored trajectory length
        if len(track.trajectory) > self.config.max_trajectory_length:
            track.trajectory = track.trajectory[-self.config.max_trajectory_length:]

        track.last_seen = timestamp
        track.visible_frames += 1
        track.missed_frames = 0
        return track

    def mark_missed(self, visible_track_ids: set[int]) -> None:
        """Increment missed frames for tracks not observed in current frame and purge expired."""
        expired_ids: List[int] = []
        for tid, track in self.active_tracks.items():
            if tid not in visible_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.config.track_buffer:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self.active_tracks[tid]

    def get_confirmed_tracks(self) -> List[TrackState]:
        """Return active tracks meeting the minimum observations threshold."""
        return [
            t for t in self.active_tracks.values()
            if t.visible_frames >= self.config.min_observations and t.missed_frames == 0
        ]
