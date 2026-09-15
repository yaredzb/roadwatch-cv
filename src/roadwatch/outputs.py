"""Structured outputs, evidence generation, and rolling frame buffers."""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from roadwatch.events import EventRecord
from roadwatch.logger import get_logger
from roadwatch.types import RiskLevel
from roadwatch.video import VideoWriter

logger = get_logger("roadwatch.outputs")


class RollingFrameBuffer:
    """Rolling buffer maintaining recent frames for pre-event clip extraction."""

    def __init__(self, max_frames: int) -> None:
        self.buffer: deque[Tuple[float, np.ndarray]] = deque(maxlen=max_frames)

    def append(self, timestamp: float, frame: np.ndarray) -> None:
        self.buffer.append((timestamp, frame.copy()))

    def get_frames(self) -> List[Tuple[float, np.ndarray]]:
        return list(self.buffer)


class EvidenceManager:
    """Manages snapshot and video clip evidence generation for detected risk events."""

    def __init__(
        self,
        output_dir: str | Path,
        fps: float,
        pre_event_sec: float = 2.0,
        post_event_sec: float = 2.0,
        enable_clips: bool = True,
        enable_snapshots: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.enable_clips = enable_clips
        self.enable_snapshots = enable_snapshots

        self.snapshots_dir = self.output_dir / "snapshots"
        self.clips_dir = self.output_dir / "clips"
        if self.enable_snapshots:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_clips:
            self.clips_dir.mkdir(parents=True, exist_ok=True)

        buffer_len = max(1, int(round(pre_event_sec * fps)))
        self.rolling_buffer = RollingFrameBuffer(max_frames=buffer_len)

        # Active clip recorders: event_id -> {frames: list, remaining_post_frames: int}
        self.pending_clips: Dict[str, Dict[str, Any]] = {}
        self.post_frames_count = int(round(post_event_sec * fps))

    def update_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Update buffer and accumulate frames for in-flight clip extractions."""
        self.rolling_buffer.append(timestamp, frame)

        # Append to pending active clips
        completed_clip_ids = []
        for evt_id, clip_data in self.pending_clips.items():
            clip_data["frames"].append(frame.copy())
            if clip_data.get("is_post_event", False):
                clip_data["remaining_post"] -= 1
                if clip_data["remaining_post"] <= 0:
                    completed_clip_ids.append(evt_id)

        for evt_id in completed_clip_ids:
            self._finalize_clip(evt_id)

    def trigger_event_start(self, event: EventRecord, current_frame: np.ndarray) -> None:
        """Save initial snapshot and start pre-event frame collection for clip."""
        evt_id = event.event_id

        # 1. Save snapshot
        if self.enable_snapshots:
            snap_path = self.snapshots_dir / f"{evt_id}.jpg"
            cv2.imwrite(str(snap_path), current_frame)
            logger.info(f"Saved snapshot: {snap_path.name}")

        # 2. Initialize clip recording
        if self.enable_clips and evt_id not in self.pending_clips:
            pre_frames = [f for _, f in self.rolling_buffer.get_frames()]
            self.pending_clips[evt_id] = {
                "frames": pre_frames,
                "is_post_event": False,
                "remaining_post": self.post_frames_count,
            }

    def trigger_event_end(self, event: EventRecord) -> None:
        """Begin post-event frame collection countdown."""
        evt_id = event.event_id
        if evt_id in self.pending_clips:
            self.pending_clips[evt_id]["is_post_event"] = True

    def _finalize_clip(self, event_id: str) -> None:
        """Encode accumulated frames into an MP4 clip."""
        clip_data = self.pending_clips.pop(event_id, None)
        if not clip_data or not clip_data["frames"]:
            return

        clip_path = self.clips_dir / f"{event_id}.mp4"
        frames = clip_data["frames"]
        h, w = frames[0].shape[:2]

        with VideoWriter(clip_path, fps=self.fps, width=w, height=h, codec="mp4v") as writer:
            for f in frames:
                writer.write_frame(f)

        logger.info(f"Saved event clip: {clip_path.name} ({len(frames)} frames)")


class OutputExporter:
    """Generates structured JSON, CSV, and summary reports."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_events_json(self, events: List[EventRecord]) -> Path:
        """Export completed events to events.json."""
        out_path = self.output_dir / "events.json"
        data = [
            {
                "event_id": e.event_id,
                "start_time": round(e.start_time, 2),
                "end_time": round(e.end_time, 2) if e.end_time is not None else None,
                "duration": round(e.end_time - e.start_time, 2) if e.end_time is not None else 0.0,
                "highest_risk_level": e.highest_risk_level.value,
                "highest_score": e.highest_score,
                "triggered_rules": sorted(list(e.triggered_rules)),
                "pedestrian_track_id": e.pedestrian_id,
                "vehicle_track_id": e.vehicle_id,
                "zone": e.zone,
                "minimum_normalized_distance": round(e.min_normalized_distance, 4),
                "snapshot": f"snapshots/{e.event_id}.jpg",
                "clip": f"clips/{e.event_id}.mp4",
            }
            for e in events
        ]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    def export_events_csv(self, events: List[EventRecord]) -> Path:
        """Export completed events to events.csv."""
        out_path = self.output_dir / "events.csv"
        fieldnames = [
            "event_id", "start_time", "end_time", "duration",
            "highest_risk_level", "highest_score", "pedestrian_track_id",
            "vehicle_track_id", "zone", "triggered_rules", "minimum_normalized_distance"
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in events:
                dur = round(e.end_time - e.start_time, 2) if e.end_time is not None else 0.0
                writer.writerow({
                    "event_id": e.event_id,
                    "start_time": round(e.start_time, 2),
                    "end_time": round(e.end_time, 2) if e.end_time is not None else None,
                    "duration": dur,
                    "highest_risk_level": e.highest_risk_level.value,
                    "highest_score": e.highest_score,
                    "pedestrian_track_id": e.pedestrian_id,
                    "vehicle_track_id": e.vehicle_id,
                    "zone": e.zone or "none",
                    "triggered_rules": ";".join(sorted(list(e.triggered_rules))),
                    "minimum_normalized_distance": round(e.min_normalized_distance, 4),
                })
        return out_path

    def export_summary_json(
        self,
        events: List[EventRecord],
        total_frames: int,
        duration: float,
        avg_fps: float,
        total_pedestrians: int,
        total_vehicles: int,
        config_path: str,
    ) -> Path:
        """Export run summary metrics to summary.json."""
        out_path = self.output_dir / "summary.json"

        by_severity = {lvl.value: 0 for lvl in RiskLevel}
        by_rule: Dict[str, int] = {}
        by_zone: Dict[str, int] = {}

        for e in events:
            by_severity[e.highest_risk_level.value] += 1
            z = e.zone or "unassigned"
            by_zone[z] = by_zone.get(z, 0) + 1
            for r in e.triggered_rules:
                by_rule[r] = by_rule.get(r, 0) + 1

        summary_data = {
            "total_frames": total_frames,
            "video_duration_seconds": round(duration, 2),
            "average_fps": round(avg_fps, 2),
            "total_tracked_pedestrians": total_pedestrians,
            "total_tracked_vehicles": total_vehicles,
            "total_events": len(events),
            "events_by_severity": by_severity,
            "events_by_rule": by_rule,
            "events_by_zone": by_zone,
            "config_file": str(config_path),
        }
        out_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        return out_path
