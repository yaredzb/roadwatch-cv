# Detailed Implementation Plan

## 1. Project Definition and Boundaries

Build a Python-based computer-vision application that analyzes recorded footage from a **fixed camera** overlooking a parking area, campus road, pedestrian crossing or similar environment.

The system must:

* Detect pedestrians and vehicles using YOLO.
* Track each object across frames.
* Monitor configurable spatial zones.
* Analyze movement and proximity.
* Identify predefined visual risk situations.
* Record evidence for detected events.
* Produce an annotated output video.
* Generate structured event and performance reports.
* Include automated tests and reproducible configuration.

### Supported object classes

Initially support:

* Person
* Bicycle
* Car
* Motorcycle
* Bus
* Truck

### Explicit limitations

The initial system must not claim to:

* Measure accurate physical distance
* Calculate exact vehicle speed
* Predict collisions
* Determine legal responsibility
* Guarantee real-world safety
* Operate reliably on moving-camera footage

It provides **visual risk indicators based on configurable rules**.

---

## 2. Code Quality and Modularity

Keep modules focused and reasonably sized. Avoid monolithic files and functions. As a guideline, production Python files should normally remain below **300 lines** and functions below **40 lines**. If a file grows beyond this, split it by responsibility unless keeping it together is demonstrably clearer. These are maintainability guidelines, not rigid limits.

Additional rules:

* Each module should have a single, clear responsibility.
* Avoid placing the entire pipeline in one large script.
* Each component should expose a clear interface (function signatures, dataclasses, or protocols).
* Shared types and constants should live in dedicated modules rather than being duplicated.
* Prefer composition over deep inheritance hierarchies.

---

## 3. System Architecture

Separate the project into independent components:

```text
Video Input
    |
    v
Frame Processor
    |
    v
YOLO Detector
    |
    v
ByteTrack Tracker
    |
    v
Trajectory Manager
    |
    v
Zone and Spatial Analyzer
    |
    v
Risk Engine
    |
    v
Event Manager
    |
    v
Visualizer and Output Writer
```

### Suggested repository structure

```text
roadwatch/
+-- configs/
|   +-- default.yaml
|   +-- example_scene.yaml
+-- data/
|   +-- raw/
|   +-- annotations/
|   +-- samples/
+-- outputs/
|   +-- videos/
|   +-- events/
|   +-- clips/
|   +-- snapshots/
|   +-- reports/
+-- scripts/
|   +-- analyze_video.py
|   +-- configure_zones.py
|   +-- evaluate_events.py
+-- src/
|   +-- roadwatch/
|       +-- __init__.py
|       +-- config.py
|       +-- video.py
|       +-- detector.py
|       +-- tracker.py
|       +-- tracks.py
|       +-- geometry.py
|       +-- zones.py
|       +-- risk_engine.py
|       +-- events.py
|       +-- visualizer.py
|       +-- outputs.py
|       +-- pipeline.py
+-- tests/
|   +-- test_config.py
|   +-- test_geometry.py
|   +-- test_zones.py
|   +-- test_trajectories.py
|   +-- test_risk_engine.py
|   +-- test_events.py
+-- .gitignore
+-- pyproject.toml
+-- README.md
+-- LICENSE
```

Videos, model weights and generated outputs should not be committed to GitHub.

---

## 4. Implementation Phases

The project is built in five sequential phases. Each phase depends on the previous one. A phase is complete only when all of its acceptance criteria are met.

---

### Phase 1: Foundation (Configuration, Video I/O, Project Structure)

**Depends on:** Nothing.

**Deliverables:**

* Repository initialized with the directory structure above.
* `pyproject.toml` with pinned dependencies.
* YAML configuration loader with full validation.
* Video reader that opens a file, extracts metadata, iterates frames, and writes an output copy.
* Structured logging setup (no `print()` calls).
* Unit tests for configuration validation.

**Acceptance criteria:**

* Running the CLI with a valid config and input video produces a byte-identical (or visually identical) output copy.
* Invalid configurations are rejected with clear, specific error messages.
* All config unit tests pass.
* The project installs cleanly from `pyproject.toml`.

**Relevant technical sections:** 5 (Configuration System), 6 (Video Input and Processing), 14 (Logging and Error Handling).

---

### Phase 2: Detection and Tracking

**Depends on:** Phase 1 (video reader, config loader).

**Deliverables:**

* YOLO detector wrapper with structured `Detection` output.
* ByteTrack integration producing persistent track IDs.
* Trajectory manager with smoothing, history limits, and expiry.
* Annotated video showing bounding boxes, class labels, track IDs, and trajectory trails.

**Acceptance criteria:**

* The detector correctly filters to supported classes and clips boxes to frame boundaries.
* Track IDs persist across consecutive frames for the same object.
* Trajectory trails render correctly on the annotated video.
* Per-frame inference time is logged.
* The system runs on both CPU and GPU (or falls back gracefully).

**Relevant technical sections:** 7 (YOLO Object Detection), 8 (Multi-Object Tracking).

---

### Phase 3: Zones, Risk Analysis, and Event Lifecycle

**Depends on:** Phase 2 (tracked objects with trajectories).

**Deliverables:**

* Zone configuration script (interactive or coordinate-based).
* Geometry utilities: point-in-polygon, line crossing, distance calculations, ground-point logic.
* Motion and pairwise interaction analysis.
* Rule-based risk engine with at least three working rules.
* Event state machine (Candidate, Active, Resolved, Cooldown).
* Stability controls: persistence, hysteresis, cooldown, track-age requirements.
* Unit tests for geometry, risk engine, trajectories, and events.

**Acceptance criteria:**

* Zones render correctly on the annotated video.
* Ground points (not box centres) determine zone membership.
* At least three risk rules trigger correctly on test footage.
* Events follow the full lifecycle without duplicate alerts.
* Cooldown prevents the same object pair from immediately re-triggering.
* All geometry, risk-engine, trajectory, and event unit tests pass.

**Relevant technical sections:** 9 (Zone Configuration and Geometry), 10 (Motion and Interaction Analysis), 11 (Rule-Based Risk Engine), 12 (Event Lifecycle Management).

---

### Phase 4: Outputs, Evidence, and CLI

**Depends on:** Phase 3 (events with risk levels).

**Deliverables:**

* Full annotated video with zones, risk overlays, timestamps, and FPS.
* Event snapshots and short video clips with pre/post context.
* Structured output files: `events.json`, `events.csv`, `summary.json`.
* Complete CLI with all options, input validation, progress display, and error codes.

**Acceptance criteria:**

* Running the full pipeline from one CLI command produces all expected outputs.
* Event clips contain context before and after the event.
* JSON and CSV outputs conform to the documented schema.
* The summary report includes frame count, duration, FPS, object counts, and event breakdowns.
* Existing output directories are not silently overwritten.

**Relevant technical sections:** 13 (Evidence and Output Generation), 14 (Command-Line Interface).

---

### Phase 5: Testing, Evaluation, and Documentation

**Depends on:** Phase 4 (complete working pipeline).

**Deliverables:**

* Integration tests using a short sample video.
* Evaluation set with manually labeled events.
* Evaluation script computing precision, recall, F1, and false-alert rate.
* Complete README with installation, usage, architecture, results, and limitations.
* At least one demonstration video or GIF.

**Acceptance criteria:**

* All unit and integration tests pass.
* The evaluation report includes quantitative metrics and qualitative failure analysis.
* The README is sufficient for a new user to install, configure, and run the system.
* Known limitations and ethical considerations are documented.
* The complete pipeline is reproducible by following the README alone.

**Relevant technical sections:** 16 (Automated Testing), 17 (Evaluation Method), 18 (Documentation Requirements).

---

## 5. Configuration System

Create a validated YAML-based configuration system so thresholds are not hardcoded.

### Configuration categories

#### Video settings

* Input path
* Output path
* Optional frame resize
* Optional frame skipping
* Output codec
* Whether to display live preview

#### Detection settings

* Model path
* Confidence threshold
* Intersection-over-Union threshold
* Relevant class names
* Inference device
* Input image size

#### Tracking settings

* Tracking algorithm
* Track confidence threshold
* Lost-track buffer
* Minimum number of observations
* Maximum stored trajectory length

#### Spatial-analysis settings

* Zone coordinates
* Line coordinates
* Distance thresholds
* Trajectory smoothing
* Movement threshold
* Closing-distance window

#### Event settings

* Minimum persistence
* Cooldown duration
* Pre-event clip duration
* Post-event clip duration
* Snapshot generation
* Enabled risk rules

### Configuration validation

The application must reject:

* Missing input files
* Invalid model paths
* Negative thresholds
* Unknown class names
* Polygons with fewer than three points
* Coordinates outside the video dimensions
* Invalid output directories
* Contradictory threshold values

Error messages should tell the user exactly what must be corrected.

---

## 6. Video Input and Processing

Implement a reusable video-processing component.

### Responsibilities

* Open the input video.

* Validate that it is readable.

* Extract:

  * Width
  * Height
  * Frame rate
  * Total frame count
  * Video duration

* Iterate through frames.

* Preserve frame timestamps.

* Optionally resize frames while maintaining aspect ratio.

* Initialize the output video writer.

* Release resources correctly after processing.

* Handle interrupted processing without corrupting existing results.

### Required behavior

The processed output must:

* Preserve the correct playback speed.
* Use the original resolution unless resizing is configured.
* Contain the same frame order as the source.
* Display a clear error if the selected codec is unavailable.

### Initial validation

Before integrating YOLO, verify that the application can read an input video and create a visually identical output copy.

---

## 7. YOLO Object Detection

Create a detector wrapper rather than calling YOLO directly throughout the codebase.

### Detector input

* Video frame
* Confidence threshold
* Allowed classes

### Detector output

Each detection should use a structured representation:

```python
Detection(
    class_id=0,
    class_name="person",
    confidence=0.91,
    bounding_box=(x1, y1, x2, y2),
    ground_point=(x_center, y2)
)
```

### Detection requirements

* Use a pretrained YOLO model initially.
* Filter detections to supported pedestrians and vehicles.
* Clip bounding boxes to the frame boundaries.
* Ignore detections below the configured threshold.
* Calculate each object's bottom-centre ground point.
* Support CPU and GPU execution.
* Measure per-frame inference time.
* Avoid fine-tuning until the pretrained baseline is evaluated.

### Visual output

Display:

* Bounding box
* Class name
* Confidence
* Distinct pedestrian and vehicle colours

### Validation

Test the detector on different scenes and document:

* Missed small objects
* Duplicate detections
* False pedestrian detections
* Performance in poor lighting
* Performance under partial occlusion

---

## 8. Multi-Object Tracking

Use ByteTrack to associate detections across consecutive frames.

### Track representation

Each active track should contain:

```python
TrackState(
    track_id=12,
    class_name="car",
    confidence=0.88,
    bounding_box=(x1, y1, x2, y2),
    ground_point=(x, y),
    smoothed_point=(x, y),
    trajectory=[...],
    first_seen=4.2,
    last_seen=8.7,
    visible_frames=96,
    missed_frames=0
)
```

### Tracking requirements

* Assign a persistent ID to each object.
* Keep pedestrian and vehicle tracks distinguishable.
* Store recent ground points.
* Limit trajectory history to avoid unlimited memory growth.
* Remove expired tracks.
* Tolerate brief missed detections.
* Handle newly appearing objects.
* Prevent invalid class changes where possible.
* Record ID switches during testing.

### Trajectory smoothing

Apply exponential smoothing or a moving average:

```text
smoothed_position =
alpha * current_position +
(1 - alpha) * previous_smoothed_position
```

The smoothing factor should be configurable.

### Visual output

Display:

* Track ID
* Recent trajectory trail
* Current movement direction
* Different colours for pedestrians and vehicles

---

## 9. Zone Configuration and Geometry

Provide a script that displays the first video frame and allows scene zones to be configured.

### Supported zone types

#### Pedestrian-only zone

Vehicles entering this zone may trigger an event.

#### Vehicle zone

Used to separate normal road activity from pedestrian areas.

#### Shared-risk zone

An area where pedestrian-vehicle interaction should be monitored closely.

#### Observation zone

Objects outside this area may be ignored.

#### Line boundary

Used for detecting crossings or movement direction.

### Zone data

Store polygons as ordered frame coordinates:

```yaml
zones:
  pedestrian_zone:
    type: pedestrian_only
    points:
      - [120, 260]
      - [640, 210]
      - [920, 620]
      - [80, 620]
```

### Geometry functions

Implement and test:

* Bounding-box centre
* Bounding-box bottom centre
* Point-in-polygon
* Line crossing
* Euclidean image distance
* Normalized image distance
* Trajectory direction
* Distance trend
* Polygon validation
* Polygon drawing

Use the ground point -- not the bounding-box centre -- to determine zone membership.

---

## 10. Motion and Interaction Analysis

Analyze each pedestrian-vehicle pair that is relevant to the same zone.

### Object motion

Calculate:

* Frame-to-frame displacement
* Smoothed displacement
* Movement direction
* Stationary versus moving state
* Recent trajectory
* Duration inside each zone

An object should be considered moving only when its displacement exceeds a configurable threshold across multiple frames.

### Pairwise interaction features

For each relevant pedestrian-vehicle pair, calculate:

* Current visual distance
* Normalized visual distance
* Minimum recent distance
* Distance change
* Whether distance is consistently decreasing
* Whether both objects occupy the same zone
* Whether their movement directions appear to converge
* Duration of the interaction

Only analyze plausible pairs. Do not compare every person with every vehicle across the entire frame if they are in unrelated zones.

### Perspective limitation

Image-space distance changes according to camera perspective. Therefore:

* Use zone-specific thresholds where necessary.
* Normalize distances by frame size or bounding-box dimensions.
* Do not label pixel distance as metres.
* Document that proximity is approximate.

Perspective calibration using homography can be added later as an extension.

---

## 11. Rule-Based Risk Engine

The risk engine must receive structured spatial and motion features. It should not depend directly on YOLO or OpenCV.

### Initial risk rules

#### Rule 1: Shared-zone conflict

Triggered when a pedestrian and moving vehicle occupy the same shared-risk zone.

#### Rule 2: Vehicle in pedestrian zone

Triggered when a moving vehicle enters a pedestrian-only area.

#### Rule 3: Close visual proximity

Triggered when a pedestrian and vehicle fall below a configured normalized-distance threshold.

#### Rule 4: Closing-distance interaction

Triggered when the distance between a pedestrian and vehicle decreases consistently across a configured sequence of frames.

#### Rule 5: Sustained exposure

Triggered when a risky interaction remains active longer than a configured duration.

### Risk scoring

Example default scoring:

| Condition                        | Score |
| -------------------------------- | ----: |
| Same shared-risk zone            |    +1 |
| Vehicle in pedestrian-only zone  |    +3 |
| Close visual proximity           |    +2 |
| Distance consistently decreasing |    +2 |
| Vehicle moving                   |    +1 |
| Interaction remains active       |    +1 |

### Risk levels

| Score | Classification |
| ----: | -------------- |
|   0-1 | Low            |
|   2-3 | Moderate       |
|   4-5 | High           |
|    6+ | Critical       |

All weights and thresholds should remain configurable.

### Stability controls

Use:

* Minimum event persistence
* Entry and exit thresholds
* Hysteresis
* Cooldown periods
* Track-age requirements
* Confidence thresholds

These controls prevent a single incorrect frame from generating an alert.

---

## 12. Event Lifecycle Management

Treat events as stateful objects rather than logging a new alert every frame.

### Event states

```text
Candidate --> Active --> Resolved --> Cooldown
```

#### Candidate

A rule has triggered, but the condition has not persisted long enough.

#### Active

The condition remained valid for the required duration.

#### Resolved

The condition is no longer present.

#### Cooldown

The same object pair cannot immediately create a duplicate event.

### Event identity

An event should be associated with:

* Pedestrian track ID
* Vehicle track ID
* Zone
* Primary risk rule

### Event record

```json
{
  "event_id": "EVT-00021",
  "start_time": 41.6,
  "end_time": 44.2,
  "highest_risk_level": "high",
  "triggered_rules": [
    "shared_zone_conflict",
    "close_proximity",
    "closing_distance"
  ],
  "pedestrian_track_id": 7,
  "vehicle_track_id": 12,
  "zone": "crossing_zone",
  "minimum_normalized_distance": 0.08,
  "snapshot": "snapshots/EVT-00021.jpg",
  "clip": "clips/EVT-00021.mp4"
}
```

---

## 13. Evidence and Output Generation

### Annotated video

The final video should display:

* Bounding boxes
* Classes and tracking IDs
* Movement trails
* Configured zones
* Active risk pairs
* Current risk level
* Triggered rules
* Video timestamp
* Processing FPS

Keep the visualization readable. Avoid covering the frame with unnecessary statistics.

### Event evidence

For active events:

* Save a clear snapshot.
* Save a short video clip containing context before and after the event.
* Highlight the relevant pedestrian and vehicle.
* Include the event ID in all filenames.

### Structured outputs

Generate:

* `events.json`
* `events.csv`
* `summary.json`
* Annotated video
* Event screenshots
* Event clips

### Summary report

Include:

* Total frames analyzed
* Video duration
* Average FPS
* Number of tracked pedestrians
* Number of tracked vehicles
* Events by severity
* Events by rule
* Events by zone
* Configuration used

---

## 14. Command-Line Interface

The project should have one main execution command:

```bash
python scripts/analyze_video.py \
  --input data/raw/scene.mp4 \
  --config configs/example_scene.yaml \
  --output outputs/run_001
```

### CLI options

Support:

* Input video
* Configuration file
* Output directory
* Model path
* Processing device
* Preview mode
* Maximum number of frames
* Disable event clips
* Verbose logging

The CLI should:

* Validate inputs before processing.
* Display progress.
* Print a final summary.
* Exit with meaningful error codes.
* Avoid replacing an existing run unless explicitly allowed.

---

## 15. Logging and Error Handling

Use structured logging rather than scattered `print()` statements.

### Log information

* Configuration loaded
* Model selected
* Processing device
* Video properties
* Output location
* Detection or tracking warnings
* Events created and resolved
* Processing completion
* Exceptions and failed output writes

Handle:

* Missing files
* Corrupted videos
* Unsupported codecs
* Empty frames
* Model-loading failure
* Invalid zone coordinates
* Unavailable GPU
* Output permission errors

If the GPU is unavailable, the program should fall back to CPU when configured to do so.

---

## 16. Automated Testing

### Unit tests

#### Geometry

* Point inside and outside polygon
* Point on boundary
* Valid and invalid polygons
* Line crossing
* Distance calculations
* Ground-point calculation

#### Trajectories

* Stationary object
* Moving object
* Increasing distance
* Decreasing distance
* Noisy positions
* Missing observations

#### Risk engine

* Individual rules
* Multiple simultaneous rules
* Correct severity calculation
* Disabled rules
* Boundary threshold values

#### Events

* Candidate activation
* Event persistence
* Event resolution
* Cooldown enforcement
* Duplicate-event prevention
* Reappearance with different track IDs

#### Configuration

* Valid configuration
* Missing required field
* Invalid threshold
* Unknown zone type
* Coordinates outside frame boundaries

### Integration tests

Use a short sample video to confirm:

* The pipeline completes successfully.
* An annotated video is created.
* Reports follow the required schema.
* Output timestamps are valid.
* Temporary resources are released.

---

## 17. Evaluation Method

Create a small evaluation set of representative video segments and manually label:

* Event start
* Event end
* Involved object types
* Zone
* Whether the event represents a visual risk condition

### Event matching

A predicted event counts as correct when:

* It involves the expected object types.
* It occurs within the labeled time interval or accepted tolerance.
* It corresponds to the correct zone.
* It is not a duplicate of an already matched prediction.

### Report

Measure:

* Event precision
* Event recall
* Event F1
* False alerts per video minute
* Duplicate alerts
* Missed events
* Average processing FPS

Also report qualitative failure cases:

* Occlusion
* Small distant pedestrians
* Shadows or reflections
* ID switches
* Camera vibration
* Poor lighting
* Perspective-related distance errors

If the pretrained detector is not fine-tuned, clearly state that the evaluation measures the **risk-event pipeline**, not a newly trained detection model.

---

## 18. Documentation Requirements

The README should contain:

* Project overview
* Problem being addressed
* Example input and output
* System architecture
* Technology stack
* Installation instructions
* Usage command
* Configuration explanation
* Risk-rule definitions
* Evaluation methodology
* Results table
* Known limitations
* Ethical and privacy considerations
* Future improvements

Also include:

* One short demonstration video or GIF
* Several event screenshots
* Example configuration
* Example event record
* Automated test instructions

---

## 19. Initial Completion Criteria

The first complete release is finished when:

* A user can supply a fixed-camera video.
* YOLO detects supported pedestrians and vehicles.
* ByteTrack maintains object identities.
* The system monitors configured zones.
* Trajectories and proximity are analyzed.
* At least three risk rules work.
* Duplicate events are controlled.
* The program exports an annotated video.
* Events are saved as JSON and CSV.
* Screenshots or clips are generated.
* Core calculations have automated tests.
* Evaluation results and limitations are documented.
* The complete pipeline can be reproduced from the README.

---

## 20. Features Not to Add Initially

* React dashboard
* Authentication
* Database
* REST API
* Cloud deployment
* Mobile application
* Live alerts
* LLM integration
* Collision prediction
* Depth estimation
* Automatic number-plate recognition
* Facial recognition
* Custom YOLO training
* Multiple-camera synchronization

Those are possible extensions, but they would distract from the central YOLO and computer-vision work.
