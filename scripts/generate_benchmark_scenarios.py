"""Synthetic scenario generator for RoadWatch benchmark testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from roadwatch.logger import setup_logger

logger = setup_logger("roadwatch.benchmark_gen")


def create_base_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a standardized roadway scene with sidewalk and zebra crossing."""
    canvas = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)

    # Road surface
    cv2.rectangle(canvas, (0, 150), (width, 450), (60, 60, 60), -1)

    # Sidewalk
    cv2.rectangle(canvas, (0, 450), (width, 480), (140, 140, 140), -1)

    # Zebra crossing stripes: x in [150, 450], y in [150, 450]
    for x in range(160, 440, 40):
        cv2.rectangle(canvas, (x, 180), (x + 20, 420), (220, 220, 220), -1)

    return canvas


def draw_actor(frame: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], size: int) -> None:
    """Render simulated actor representation."""
    x, y = center
    cv2.circle(frame, (x, y), size, color, -1)
    cv2.circle(frame, (x, y), size, (255, 255, 255), 1)


def generate_crossing_conflict_video(output_path: Path, fps: float = 20.0, duration_sec: float = 4.0) -> None:
    """Generate scenario where vehicle and pedestrian cross paths in zebra crossing."""
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

    base = create_base_scene()
    for f in range(total_frames):
        t = f / total_frames
        frame = base.copy()

        # Vehicle moves horizontally across road: x from 20 to 600, y = 300
        veh_x = int(20 + t * 580)
        veh_y = 300
        draw_actor(frame, (veh_x, veh_y), (40, 140, 240), size=24) # Vehicle

        # Pedestrian walks vertically downwards across zebra crossing: x = 300, y from 180 to 450
        ped_x = 300
        ped_y = int(180 + t * 270)
        draw_actor(frame, (ped_x, ped_y), (240, 160, 50), size=12) # Pedestrian

        out.write(frame)

    out.release()
    logger.info(f"Generated crossing conflict scenario: {output_path} ({total_frames} frames)")


def generate_safe_passage_video(output_path: Path, fps: float = 20.0, duration_sec: float = 4.0) -> None:
    """Generate negative control scenario where vehicle passes completely before pedestrian crosses."""
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

    base = create_base_scene()
    half = total_frames // 2

    for f in range(total_frames):
        frame = base.copy()
        if f < half:
            # First half: Vehicle travels through road
            t = f / half
            veh_x = int(20 + t * 580)
            draw_actor(frame, (veh_x, 300), (40, 140, 240), size=24)
        else:
            # Second half: Vehicle gone, pedestrian begins crossing safely
            t = (f - half) / half
            ped_y = int(180 + t * 270)
            draw_actor(frame, (300, ped_y), (240, 160, 50), size=12)

        out.write(frame)

    out.release()
    logger.info(f"Generated safe passage scenario: {output_path} ({total_frames} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Benchmark Synthetic Scenarios")
    parser.add_argument("-o", "--output-dir", type=str, default="data/samples", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conflict_path = out_dir / "scenario_crossing_conflict.mp4"
    safe_path = out_dir / "scenario_safe_passage.mp4"

    generate_crossing_conflict_video(conflict_path)
    generate_safe_passage_video(safe_path)

    logger.info("All benchmark scenarios created successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
