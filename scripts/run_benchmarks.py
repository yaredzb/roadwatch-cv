"""Automated benchmark execution and evaluation runner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roadwatch.config import load_config
from roadwatch.logger import setup_logger
from scripts.analyze_video import process_video_stream
from scripts.evaluate_events import compute_metrics, match_events

logger = setup_logger("roadwatch.benchmark_runner")


def run_benchmark_suite(
    config_path: Path,
    ground_truth_path: Path,
    output_base_dir: Path,
) -> Dict[str, Any]:
    """Execute complete pipeline across all benchmark scenarios and aggregate metrics."""
    cfg = load_config(config_path)
    gt_data = json.loads(ground_truth_path.read_text(encoding="utf-8"))

    scenarios = [
        "scenario_crossing_conflict",
        "scenario_safe_passage",
    ]

    results: Dict[str, Any] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_dups = 0
    total_duration = 0.0

    print("\n" + "=" * 60)
    print("        ROADWATCH AUTOMATED BENCHMARK SUITE")
    print("=" * 60)

    for sc_name in scenarios:
        video_file = PROJECT_ROOT / "data" / "samples" / f"{sc_name}.mp4"
        if not video_file.exists():
            logger.warning(f"Scenario video not found: {video_file}. Skipping.")
            continue

        sc_out_dir = output_base_dir / sc_name
        sc_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running scenario: {sc_name}...")
        process_video_stream(
            cfg=cfg,
            in_file=video_file,
            out_dir=sc_out_dir,
            max_frames=None,
            disable_clips=True,
            logger=logger,
        )

        pred_file = sc_out_dir / "events.json"
        preds = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.exists() else []

        sum_file = sc_out_dir / "summary.json"
        summary = json.loads(sum_file.read_text(encoding="utf-8")) if sum_file.exists() else {}
        sc_duration = float(summary.get("video_duration_seconds", 4.0))
        total_duration += sc_duration

        # Filter GT relevant to this scenario
        sc_gt = [g for g in gt_data if g.get("scenario") == sc_name]
        tp, fp, fn, dups = match_events(preds, sc_gt, time_tolerance=2.0)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_dups += dups

        sc_metrics = compute_metrics(tp, fp, fn, dups, sc_duration)
        sc_metrics["average_fps"] = summary.get("average_fps", 0.0)
        results[sc_name] = sc_metrics

        print(f"Scenario [{sc_name}]:")
        print(f"  GT: {len(sc_gt)} | Preds: {len(preds)} | TP: {tp} | FP: {fp} | FN: {fn} | FPS: {summary.get('average_fps', 0.0):.1f}")

    overall_metrics = compute_metrics(total_tp, total_fp, total_fn, total_dups, total_duration)
    results["overall_summary"] = overall_metrics

    # Save summary report
    report_file = output_base_dir / "benchmark_report.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("-" * 60)
    print(" OVERALL BENCHMARK RESULTS")
    print("-" * 60)
    print(f" Total True Positives  : {total_tp}")
    print(f" Total False Positives : {total_fp}")
    print(f" Total False Negatives : {total_fn}")
    print(f" Overall Precision     : {overall_metrics['precision']:.2%}")
    print(f" Overall Recall        : {overall_metrics['recall']:.2%}")
    print(f" Overall F1-Score      : {overall_metrics['f1_score']:.2%}")
    print(f" False Alert Rate / min: {overall_metrics['false_alerts_per_minute']:.2f}")
    print("=" * 60)
    logger.info(f"Saved benchmark report: {report_file}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoadWatch Benchmark Suite")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("-g", "--ground-truth", type=str, default="data/annotations/ground_truth_events.json")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs/benchmark_results")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.output_dir)

    if not cfg_path.exists() or not gt_path.exists():
        logger.error("Required configuration or ground truth file missing.")
        return 1

    run_benchmark_suite(cfg_path, gt_path, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
