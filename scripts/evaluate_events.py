"""Evaluation script for benchmark event performance against ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from roadwatch.logger import setup_logger

logger = setup_logger("roadwatch.evaluate")


def match_events(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    time_tolerance: float = 2.0,
) -> Tuple[int, int, int, int]:
    """
    Match predictions against ground truth events.
    Returns (True Positives, False Positives, False Negatives, Duplicates).
    """
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    duplicates = 0

    for pred in predictions:
        p_start = float(pred["start_time"])
        p_end = float(pred.get("end_time") or p_start)
        p_zone = pred.get("zone")

        # Find best matching GT event
        best_match_idx: Optional[int] = None
        for i, gt in enumerate(ground_truth):
            gt_start = float(gt["start_time"])
            gt_end = float(gt.get("end_time") or gt_start)
            gt_zone = gt.get("zone")

            # Check temporal proximity / overlap
            time_overlap = not (p_end < (gt_start - time_tolerance) or p_start > (gt_end + time_tolerance))
            zone_match = (gt_zone is None) or (p_zone == gt_zone)

            if time_overlap and zone_match:
                best_match_idx = i
                break

        if best_match_idx is not None:
            if best_match_idx in matched_gt:
                duplicates += 1
                fp += 1  # Duplicate counts as extra alert
            else:
                matched_gt.add(best_match_idx)
                tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn, duplicates


def compute_metrics(
    tp: int,
    fp: int,
    fn: int,
    duplicates: int,
    duration_sec: float,
) -> Dict[str, float]:
    """Calculate Precision, Recall, F1-score, and False Alert Rate per minute."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    duration_min = max(0.01, duration_sec / 60.0)
    false_alerts_per_min = fp / duration_min

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "duplicate_alerts": duplicates,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_alerts_per_minute": round(false_alerts_per_min, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Predicted Events against Ground Truth")
    parser.add_argument("-p", "--predictions", type=str, required=True, help="Path to events.json")
    parser.add_argument("-g", "--ground-truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("-s", "--summary", type=str, default=None, help="Path to summary.json (for duration)")
    parser.add_argument("-o", "--output", type=str, default="evaluation_report.json", help="Report output path")
    parser.add_argument("-t", "--tolerance", type=float, default=2.0, help="Time tolerance in seconds")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)
    if not pred_path.exists() or not gt_path.exists():
        logger.error(f"Missing input files: {pred_path} or {gt_path}")
        return 1

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    duration = 60.0
    if args.summary and Path(args.summary).exists():
        sum_data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        duration = float(sum_data.get("video_duration_seconds", 60.0))

    tp, fp, fn, dups = match_events(preds, gt, time_tolerance=args.tolerance)
    metrics = compute_metrics(tp, fp, fn, dups, duration)

    report_path = Path(args.output)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 45)
    print("       ROADWATCH EVENT EVALUATION REPORT     ")
    print("=" * 45)
    print(f" Ground Truth Events : {len(gt)}")
    print(f" Predicted Events    : {len(preds)}")
    print(f" True Positives (TP) : {tp}")
    print(f" False Positives(FP) : {fp}")
    print(f" False Negatives(FN) : {fn}")
    print(f" Duplicate Alerts    : {dups}")
    print("-" * 45)
    print(f" Precision           : {metrics['precision']:.2%}")
    print(f" Recall              : {metrics['recall']:.2%}")
    print(f" F1-Score            : {metrics['f1_score']:.2%}")
    print(f" False Alerts / Min  : {metrics['false_alerts_per_minute']:.2f}")
    print("=" * 45)
    logger.info(f"Evaluation report saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
