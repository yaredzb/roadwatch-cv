"""Main execution script for RoadWatch video analysis."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add src/ to sys.path so the package can be executed directly without pip install -e .
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from roadwatch.config import AppConfig, load_config
from roadwatch.detector import YOLODetector
from roadwatch.events import EventManager
from roadwatch.geometry import euclidean_distance
from roadwatch.logger import setup_logger
from roadwatch.outputs import EvidenceManager, OutputExporter
from roadwatch.risk_engine import RiskEngine
from roadwatch.tracker import ObjectTracker
from roadwatch.tracks import TrajectoryManager
from roadwatch.video import VideoReader, VideoWriter
from roadwatch.visualizer import Visualizer
from roadwatch.zones import ZoneManager


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RoadWatch: Pedestrian-Vehicle Risk Monitoring System"
    )
    parser.add_argument("-i", "--input", type=str, help="Path to input video file")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("-o", "--output", type=str, help="Path to output directory")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration file syntax and exit",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--disable-clips",
        action="store_true",
        help="Disable generating video clips for events",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug/verbose logging"
    )
    return parser.parse_args()


def process_video_stream(
    cfg: AppConfig,
    in_file: Path,
    out_dir: Path,
    max_frames: int | None,
    disable_clips: bool,
    logger,
) -> None:
    """Execute full video analysis and output generation."""
    out_video_path = out_dir / f"annotated_{in_file.name}"
    detector = YOLODetector(cfg.detection)
    tracker = ObjectTracker(cfg.tracking)
    traj_manager = TrajectoryManager(
        cfg.tracking, smoothing_alpha=cfg.spatial.trajectory_smoothing_alpha
    )
    zone_mgr = ZoneManager(cfg.spatial)
    risk_engine = RiskEngine(cfg.events, cfg.spatial)
    event_mgr = EventManager(cfg.events)
    visualizer = Visualizer(show_trails=True)
    exporter = OutputExporter(out_dir)

    all_pedestrian_ids = set()
    all_vehicle_ids = set()
    pair_distances: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_starts: dict[tuple[int, int], float] = {}

    start_time = time.time()
    frames_processed = 0

    with VideoReader(in_file) as reader:
        meta = reader.metadata
        evidence = EvidenceManager(
            output_dir=out_dir,
            fps=meta.fps,
            pre_event_sec=cfg.events.pre_event_clip_sec,
            post_event_sec=cfg.events.post_event_clip_sec,
            enable_clips=(not disable_clips and cfg.events.pre_event_clip_sec > 0),
            enable_snapshots=cfg.events.snapshot_generation,
        )

        with VideoWriter(
            output_path=out_video_path,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            codec=cfg.video.output_codec,
        ) as writer:
            for frame_idx, timestamp, frame in reader.read_frames(
                frame_skip=cfg.video.frame_skip, max_frames=max_frames
            ):
                frame_t0 = time.perf_counter()

                # Detect & Track
                detections, _ = detector.detect(frame)
                tracked_results = tracker.update(detections)

                visible_ids = set()
                for tid, det in tracked_results:
                    visible_ids.add(tid)
                    traj_manager.update_track(
                        track_id=tid, class_name=det.class_name, confidence=det.confidence,
                        bounding_box=det.bounding_box, timestamp=timestamp,
                    )
                    if det.class_name == "person":
                        all_pedestrian_ids.add(tid)
                    else:
                        all_vehicle_ids.add(tid)

                traj_manager.mark_missed(visible_ids)
                tracks = traj_manager.get_confirmed_tracks()

                # Evaluate Risk
                peds = [t for t in tracks if t.is_pedestrian]
                vehs = [t for t in tracks if t.is_vehicle]
                assessments = []
                for p in peds:
                    for v in vehs:
                        pair = (p.track_id, v.track_id)
                        pair_distances[pair].append(euclidean_distance(p.smoothed_point, v.smoothed_point))
                        if pair not in pair_starts:
                            pair_starts[pair] = timestamp
                        a = risk_engine.evaluate_pair(
                            ped=p, veh=v, distance_history=pair_distances[pair],
                            zone_mgr=zone_mgr, frame_width=meta.width, frame_height=meta.height,
                            interaction_duration=(timestamp - pair_starts[pair]),
                        )
                        assessments.append(a)

                # Event Lifecycle & Evidence
                activated, resolved = event_mgr.update(assessments, timestamp)
                for act in activated:
                    evidence.trigger_event_start(act, frame)
                for res in resolved:
                    evidence.trigger_event_end(res)

                evidence.update_frame(timestamp, frame)

                # Render & Write
                frame_fps = 1.0 / max(1e-5, time.perf_counter() - frame_t0)
                annotated = visualizer.render(
                    frame=frame, tracks=tracks, frame_idx=frame_idx,
                    timestamp=timestamp, fps=frame_fps, zone_mgr=zone_mgr,
                    assessments=assessments,
                )
                writer.write_frame(annotated)
                frames_processed += 1

    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    events = event_mgr.completed_events

    # Export Structured Reports
    exporter.export_events_json(events)
    exporter.export_events_csv(events)
    exporter.export_summary_json(
        events=events, total_frames=frames_processed, duration=total_time,
        avg_fps=avg_fps, total_pedestrians=len(all_pedestrian_ids),
        total_vehicles=len(all_vehicle_ids), config_path=str(cfg.video.input_path),
    )

    logger.info(
        f"Processing complete: {frames_processed} frames in {total_time:.2f}s (Avg {avg_fps:.1f} FPS) | "
        f"Detected {len(events)} events | Outputs saved to {out_dir}"
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_arguments()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("roadwatch", log_level=log_level)

    try:
        if args.validate_config:
            logger.info(f"Validating configuration at: {args.config}")
            load_config(args.config)
            logger.info("Configuration is VALID.")
            return 0

        cfg = load_config(args.config)
        in_file = Path(args.input or cfg.video.input_path)
        out_dir = Path(args.output or cfg.video.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        process_video_stream(
            cfg=cfg, in_file=in_file, out_dir=out_dir,
            max_frames=args.max_frames, disable_clips=args.disable_clips,
            logger=logger,
        )
        return 0

    except Exception as exc:
        logger.error(f"Execution failed: {exc}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
