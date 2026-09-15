"""Interactive polygon zone configuration tool for RoadWatch."""

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

from roadwatch.logger import setup_logger
from roadwatch.types import Point, ZoneType
from roadwatch.video import VideoReader

logger = setup_logger("roadwatch.configure_zones")


class ZoneConfigurator:
    """Interactive GUI for defining polygon zones on the initial video frame."""

    def __init__(self, frame: np.ndarray, config_path: Path) -> None:
        self.base_frame = frame.copy()
        self.config_path = config_path
        self.current_points: List[Tuple[int, int]] = []
        self.saved_zones: dict[str, dict] = {}
        self.window_name = "RoadWatch Zone Configurator"

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """Record clicked vertices."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            logger.info(f"Added vertex: ({x}, {y})")

    def _render_display(self) -> np.ndarray:
        """Draw existing zones and active polygon in progress."""
        canvas = self.base_frame.copy()

        # Draw already saved zones
        for name, data in self.saved_zones.items():
            pts = np.array(data["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cx = int(np.mean([p[0] for p in data["points"]]))
            cy = int(np.mean([p[1] for p in data["points"]]))
            cv2.putText(
                canvas, f"{name} ({data['type']})", (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Draw current polygon vertices
        for pt in self.current_points:
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1)

        if len(self.current_points) >= 2:
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 200, 255), thickness=1)

        return canvas

    def run_interactive(self) -> None:
        """Main interaction event loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Zone Configurator Controls ===")
        print(" Left Click : Add polygon vertex")
        print(" 'c'        : Close current polygon & assign zone")
        print(" 'r'        : Reset current in-progress points")
        print(" 's'        : Save all zones to YAML config and exit")
        print(" 'q'        : Quit without saving\n")

        while True:
            display = self._render_display()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_points.clear()
                logger.info("Current points reset.")
            elif key == ord('c'):
                self._prompt_and_add_zone()
            elif key == ord('s'):
                self._save_to_yaml()
                break

        cv2.destroyAllWindows()

    def _prompt_and_add_zone(self) -> None:
        """Prompt user for zone name and type in terminal."""
        if len(self.current_points) < 3:
            logger.warning("A polygon must have at least 3 points before closing.")
            return

        name = input("Enter zone name (e.g. crosswalk): ").strip()
        print("Select zone type: 1) shared_risk  2) pedestrian_only  3) vehicle_zone")
        choice = input("Enter number [1]: ").strip() or "1"
        type_map = {"1": "shared_risk", "2": "pedestrian_only", "3": "vehicle_zone"}
        z_type = type_map.get(choice, "shared_risk")

        self.saved_zones[name] = {
            "type": z_type,
            "points": [list(pt) for pt in self.current_points]
        }
        logger.info(f"Recorded zone '{name}' ({z_type}) with {len(self.current_points)} vertices.")
        self.current_points.clear()

    def _save_to_yaml(self) -> None:
        """Persist zones back into YAML configuration."""
        if not self.saved_zones:
            logger.warning("No zones defined to save.")
            return

        config_data: dict = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        if "spatial" not in config_data:
            config_data["spatial"] = {}
        if "zones" not in config_data["spatial"]:
            config_data["spatial"]["zones"] = {}

        config_data["spatial"]["zones"].update(self.saved_zones)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)

        logger.info(f"Successfully saved {len(self.saved_zones)} zones into: {self.config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RoadWatch: Interactive Zone Configurator")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video path")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml", help="Target config YAML")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    with VideoReader(video_path) as reader:
        for _, _, frame in reader.read_frames(max_frames=1):
            configurator = ZoneConfigurator(frame, Path(args.config))
            configurator.run_interactive()
            return 0

    logger.error("Could not read initial frame from video.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
