"""Interactive camera perspective calibration tool for RoadWatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import yaml

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from roadwatch.calibration import PerspectiveCalibrator
from roadwatch.logger import setup_logger
from roadwatch.video import VideoReader

logger = setup_logger("roadwatch.calibrate")


class CameraCalibratorGUI:
    """Interactive GUI for selecting 4 planar reference points on video frame."""

    def __init__(self, frame: np.ndarray, config_path: Path) -> None:
        self.base_frame = frame.copy()
        self.config_path = config_path
        self.points: List[Tuple[int, int]] = []
        self.window_name = "RoadWatch Perspective Calibrator"

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """Collect clicked reference points."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            logger.info(f"Point {len(self.points)}/4: ({x}, {y})")

    def _render(self) -> np.ndarray:
        canvas = self.base_frame.copy()
        for i, pt in enumerate(self.points):
            cv2.circle(canvas, pt, 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"P{i+1}", (pt[0] + 8, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            closed = len(self.points) == 4
            cv2.polylines(canvas, [pts], isClosed=closed, color=(0, 255, 255), thickness=2)

        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Perspective Calibrator Instructions ===")
        print(" Click 4 points in clockwise order defining a flat rectangular area:")
        print(" 1: Top-Left  2: Top-Right  3: Bottom-Right  4: Bottom-Left")
        print(" 'r' : Reset points")
        print(" 'q' : Cancel\n")

        while True:
            display = self._render()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points.clear()
                logger.info("Points reset.")

            if len(self.points) == 4:
                cv2.imshow(self.window_name, self._render())
                cv2.waitKey(200)
                self._prompt_and_save()
                break

        cv2.destroyAllWindows()

    def _prompt_and_save(self) -> None:
        print("\n4 points recorded. Enter real-world physical dimensions:")
        try:
            w_str = input("Real-world Width in meters (between P1 and P2, e.g. 5.0): ").strip()
            l_str = input("Real-world Length in meters (between P2 and P3, e.g. 8.0): ").strip()
            width_m = float(w_str)
            length_m = float(l_str)
        except ValueError:
            logger.error("Invalid numeric input for physical dimensions.")
            return

        # Validate homography
        try:
            calib = PerspectiveCalibrator(
                source_points=[(float(p[0]), float(p[1])) for p in self.points],
                target_dimensions_meters=(width_m, length_m),
            )
        except Exception as exc:
            logger.error(f"Homography validation failed: {exc}")
            return

        config_data = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        config_data["calibration"] = {
            "enabled": True,
            "source_points": [list(p) for p in self.points],
            "target_dimensions_meters": [width_m, length_m],
            "metric_distance_threshold": 2.5,
            "speed_threshold_kmh": 20.0,
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)

        logger.info(f"Saved calibration into {self.config_path} successfully!")


def main() -> int:
    parser = argparse.ArgumentParser(description="RoadWatch: Perspective Calibrator")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video path")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml", help="Target config YAML")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1

    with VideoReader(video_path) as reader:
        for _, _, frame in reader.read_frames(max_frames=1):
            gui = CameraCalibratorGUI(frame, Path(args.config))
            gui.run()
            return 0

    logger.error("Failed to read video frame.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
