"""ByteTrack adapter using supervision."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import supervision as sv

from roadwatch.config import TrackingConfig
from roadwatch.logger import get_logger
from roadwatch.types import BoundingBox, Detection

logger = get_logger("roadwatch.tracker")


class ObjectTracker:
    """Wraps ByteTrack algorithm to associate detections across consecutive frames."""

    def __init__(self, config: TrackingConfig, frame_rate: int = 30) -> None:
        self.config = config
        self.frame_rate = frame_rate

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.config.track_thresh,
                lost_track_buffer=self.config.track_buffer,
                minimum_matching_threshold=self.config.match_thresh,
                frame_rate=self.frame_rate,
            )
        logger.info(
            f"Initialized ByteTrack (thresh={self.config.track_thresh}, "
            f"buffer={self.config.track_buffer}, match={self.config.match_thresh})"
        )

    def update(
        self, detections: List[Detection]
    ) -> List[Tuple[int, Detection]]:
        """
        Update tracker with current frame detections.
        Returns list of (track_id, detection) pairs for successfully tracked objects.
        """
        if not detections:
            # Update tracker with empty detections to advance internal Kalman filters
            empty_dets = sv.Detections.empty()
            self.byte_tracker.update_with_detections(empty_dets)
            return []

        # Convert Detection list into supervision Detections object
        xyxy = np.array([d.bounding_box.as_xyxy() for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_sv = self.byte_tracker.update_with_detections(sv_detections)

        tracked_results: List[Tuple[int, Detection]] = []
        if tracked_sv.tracker_id is None:
            return tracked_results

        # Reconstruct structured Detection objects with persistent track IDs
        for i, tid in enumerate(tracked_sv.tracker_id):
            if tid is None:
                continue

            box = BoundingBox(
                x1=float(tracked_sv.xyxy[i][0]),
                y1=float(tracked_sv.xyxy[i][1]),
                x2=float(tracked_sv.xyxy[i][2]),
                y2=float(tracked_sv.xyxy[i][3]),
            )
            cls_id = int(tracked_sv.class_id[i])
            conf = float(tracked_sv.confidence[i])

            # Find matching original class name
            matching_det = next((d for d in detections if d.class_id == cls_id), None)
            cls_name = matching_det.class_name if matching_det else "object"

            detection = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bounding_box=box,
                ground_point=box.bottom_center,
            )
            tracked_results.append((int(tid), detection))

        return tracked_results
