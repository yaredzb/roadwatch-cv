"""Object detector wrapper for Ultralytics YOLO."""

from __future__ import annotations

import time
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

from roadwatch.config import DetectionConfig
from roadwatch.logger import get_logger
from roadwatch.types import BoundingBox, Detection, Point

logger = get_logger("roadwatch.detector")


class YOLODetector:
    """Wraps YOLO inference, class filtering, and bounding box clipping."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model_path = config.model_path
        self.conf_thresh = config.confidence_threshold
        self.iou_thresh = config.iou_threshold
        self.device = config.device
        self.img_size = config.img_size
        self.relevant_classes = set(c.lower() for c in config.relevant_classes)

        logger.info(f"Loading YOLO model from '{self.model_path}' on device '{self.device}'...")
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded. Monitoring classes: {sorted(self.relevant_classes)}")

    def _clip_box(self, box: Tuple[float, float, float, float], width: int, height: int) -> BoundingBox:
        """Ensure bounding box coordinates remain strictly within frame bounds."""
        x1, y1, x2, y2 = box
        clipped_x1 = max(0.0, min(float(x1), float(width)))
        clipped_y1 = max(0.0, min(float(y1), float(height)))
        clipped_x2 = max(clipped_x1, min(float(x2), float(width)))
        clipped_y2 = max(clipped_y1, min(float(y2), float(height)))
        return BoundingBox(x1=clipped_x1, y1=clipped_y1, x2=clipped_x2, y2=clipped_y2)

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run inference on a single BGR frame.
        Returns (list_of_detections, inference_time_seconds).
        """
        start_t = time.perf_counter()
        height, width = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            imgsz=self.img_size,
            verbose=False,
        )

        inference_time = time.perf_counter() - start_t
        detections: List[Detection] = []

        if not results:
            return detections, inference_time

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections, inference_time

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = self.model.names

        for box_coords, conf, cls_id in zip(boxes, confs, class_ids):
            class_name = names.get(cls_id, f"class_{cls_id}").lower()
            if class_name not in self.relevant_classes:
                continue

            clipped_box = self._clip_box(box_coords, width=width, height=height)
            ground_pt = clipped_box.bottom_center

            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bounding_box=clipped_box,
                    ground_point=ground_pt,
                )
            )

        return detections, inference_time
