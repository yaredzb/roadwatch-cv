# RoadWatch: Pedestrian–Vehicle Risk Monitoring System

RoadWatch is a Python-based computer-vision application designed to monitor and identify visual risk situations between pedestrians and vehicles from **fixed-camera** video footage (e.g. campus crossings, parking areas, and shared roadways).

---

## 1. System Architecture

```text
Video Stream (Input)
        |
        v
VideoReader & Rolling Frame Buffer
        |
        v
YOLO Detector (Ultralytics)
        |
        v
ByteTrack Multi-Object Tracker
        |
        v
Trajectory Manager & Smoothing
        |
        v
Spatial Zone Analyzer (Ground Point In-Polygon)
        |
        v
Rule-Based Risk Assessment Engine
        |
        v
Event Lifecycle State Machine (Candidate -> Active -> Resolved -> Cooldown)
        |
        v
Visualizer & Output Exporter
  +-- Annotated Output Video (MP4)
  +-- Event Snapshots & Video Clips
  +-- Structured Reports (events.json, events.csv, summary.json)
```

---

## 2. Project Structure

```text
roadwatch/
├── configs/
│   └── default.yaml             # Master validated configuration
├── data/
│   ├── raw/                     # Raw input videos (ignored)
│   ├── annotations/             # Ground truth benchmark annotations
│   └── samples/                 # Sample video clips
├── outputs/                     # Generated videos, clips, snapshots, reports (ignored)
├── scripts/
│   ├── analyze_video.py         # Main analysis CLI entrypoint
│   ├── configure_zones.py       # Interactive polygon zone configuration tool
│   └── evaluate_events.py       # Event precision/recall/F1 benchmarking tool
├── src/
│   └── roadwatch/
│       ├── __init__.py
│       ├── types.py             # Shared dataclasses and Enums
│       ├── logger.py            # Structured logging handler
│       ├── config.py            # Pydantic-validated YAML parser
│       ├── video.py             # VideoReader and VideoWriter abstractions
│       ├── detector.py          # YOLO detector wrapper
│       ├── tracker.py           # ByteTrack adapter
│       ├── tracks.py            # Trajectory state and exponential smoothing
│       ├── geometry.py          # Point-in-polygon, distances, trend detection
│       ├── zones.py             # Spatial zone querying and rendering
│       ├── risk_engine.py       # Rule-based risk scoring engine
│       ├── events.py            # Stateful event lifecycle manager
│       ├── visualizer.py        # Video annotation and HUD overlay
│       └── outputs.py           # Rolling buffer, clips, and structured exporters
├── tests/                       # Complete pytest unit and integration test suite
├── pyproject.toml               # Package dependencies and settings
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
- Python >= 3.10
- GPU (CUDA) recommended for real-time inference; CPU fully supported.

### Install Dependencies
```bash
git clone https://github.com/yaredzb/roadwatch-cv.git
cd "Pedestrian–Vehicle Risk Monitoring System"

# Install package dependencies
pip install -e .
```

---

## 4. Usage Guide

### 4.1 Running Video Analysis
To analyze a video using the default configuration:
```bash
python scripts/analyze_video.py \
  --input data/samples/sample.mp4 \
  --config configs/default.yaml \
  --output outputs/run_001
```

**CLI Options**:
- `-i, --input`: Path to input video file (overrides config).
- `-c, --config`: Path to YAML configuration file (default: `configs/default.yaml`).
- `-o, --output`: Output directory where video and reports will be saved.
- `--validate-config`: Validates configuration syntax and exits without running video.
- `--max-frames`: Process only the first $N$ frames (useful for quick verification).
- `--disable-clips`: Disable generating individual MP4 video clips for events.
- `-v, --verbose`: Enable debug logging.

### 4.2 Interactive Zone Configuration
Define polygon zones interactively on the initial video frame:
```bash
python scripts/configure_zones.py --input data/samples/sample.mp4 --config configs/default.yaml
```
**Controls**:
- **Left Click**: Place polygon vertex.
- **`c`**: Close current polygon and select zone type (`shared_risk`, `pedestrian_only`, `vehicle_zone`).
- **`r`**: Reset current in-progress polygon.
- **`s`**: Save all zones back into YAML configuration.
- **`q`**: Quit without saving.

### 4.3 Benchmarking & Event Evaluation
Evaluate detected events against ground truth annotations:
```bash
python scripts/evaluate_events.py \
  --predictions outputs/run_001/events.json \
  --ground-truth data/annotations/ground_truth.json \
  --summary outputs/run_001/summary.json \
  --output outputs/run_001/evaluation_report.json
```

---

## 5. Risk Assessment Engine & Rules

The system evaluates pairwise interactions using object ground contact points (bottom-center of bounding box):

| Condition | Default Score | Description |
| :--- | :---: | :--- |
| **Shared-Zone Conflict** | +1 | Pedestrian and moving vehicle occupy the same `shared_risk` zone. |
| **Vehicle in Pedestrian Zone** | +3 | Moving vehicle enters a `pedestrian_only` area. |
| **Close Proximity** | +2 | Normalized visual distance drops below configured threshold. |
| **Closing-Distance Interaction** | +2 | Pairwise distance consistently decreases across consecutive observations. |
| **Sustained Exposure** | +1 | Interaction remains active longer than duration threshold. |

### Severity Categorization
- **Low**: 0 – 1 points
- **Moderate**: 2 – 3 points
- **High**: 4 – 5 points
- **Critical**: 6+ points

---

## 6. Event Lifecycle State Machine

Events are managed as stateful objects to prevent duplicate alerts and false flickering:
```text
[Candidate]  --> (persisted >= min_persistence_sec) --> [Active]
     |                                                      |
(clears before persistence)                       (interaction ends)
     |                                                      v
[Dropped]                                              [Resolved]
                                                            |
                                                            v
                                                       [Cooldown]
```
- **Candidate**: Risk condition detected; timers start.
- **Active**: Condition persisted continuously $\ge 0.5$s; snapshot captured, clip recording triggered.
- **Resolved**: Object separated or exited danger zone; event finalized.
- **Cooldown**: Minimum period (default 3.0s) preventing duplicate alerts for the same object pair.

---

## 7. Automated Testing

The repository contains automated unit and integration tests covering geometry math, configuration validation, tracker state, risk rules, and export schemas:
```bash
pytest tests/ -v
```

---

## 8. Limitations & Ethical Considerations

### Explicit Limitations
- **Camera Rigidity**: Designed for fixed surveillance cameras; not calibrated for moving/panning cameras.
- **Approximate Proximity**: Proximity measurements are computed in normalized image space, not real-world metric distances.
- **No Collision Guarantees**: Intended as visual indicators and risk analytics; does not predict physical collisions or determine legal responsibility.

### Privacy & Ethical Principles
- Does **not** perform facial recognition or automated license plate recognition (ALPR).
- Uses generalized object classes (`person`, `car`, `bus`, `truck`, `bicycle`, `motorcycle`) to respect individual privacy.
