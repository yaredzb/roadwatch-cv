"""Tests for video input, output, and metadata handling."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from roadwatch.video import VideoReader, VideoWriter


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """Create a temporary 10-frame synthetic test video (640x480 @ 10 FPS)."""
    video_path = tmp_path / "test_synthetic.mp4"
    width, height, fps, num_frames = 320, 240, 10.0, 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Frame with changing color and frame index
        frame = np.full((height, width, 3), (i * 20, 120, 200), dtype=np.uint8)
        out.write(frame)

    out.release()
    return video_path


def test_video_reader_metadata(synthetic_video):
    """Verify video reader accurately extracts resolution, FPS, duration, and frame count."""
    with VideoReader(synthetic_video) as reader:
        meta = reader.metadata
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == pytest.approx(10.0, abs=0.5)
        assert meta.total_frames == 10
        assert meta.duration == pytest.approx(1.0, abs=0.1)


def test_video_reader_frames(synthetic_video):
    """Verify that frames are iterated with consistent sequential indices and timestamps."""
    with VideoReader(synthetic_video) as reader:
        frames = list(reader.read_frames())
        assert len(frames) == 10

        for expected_idx, (frame_idx, timestamp, frame) in enumerate(frames):
            assert frame_idx == expected_idx
            assert frame.shape == (240, 320, 3)
            assert timestamp >= 0.0


def test_video_reader_max_frames(synthetic_video):
    """Verify max_frames parameter stops iteration early."""
    with VideoReader(synthetic_video) as reader:
        frames = list(reader.read_frames(max_frames=4))
        assert len(frames) == 4


def test_video_writer_roundtrip(synthetic_video, tmp_path):
    """Verify writing frames produces a valid, readable output video."""
    out_video_path = tmp_path / "output_roundtrip.mp4"

    with VideoReader(synthetic_video) as reader:
        meta = reader.metadata
        with VideoWriter(out_video_path, fps=meta.fps, width=meta.width, height=meta.height) as writer:
            for _, _, frame in reader.read_frames():
                writer.write_frame(frame)

    assert out_video_path.exists()
    assert out_video_path.stat().st_size > 0

    # Read back the created video
    with VideoReader(out_video_path) as read_back:
        assert read_back.metadata.width == 320
        assert read_back.metadata.height == 240
        read_frames = list(read_back.read_frames())
        assert len(read_frames) == 10


def test_video_reader_missing_file():
    """Verify VideoReader raises FileNotFoundError when video is absent."""
    with pytest.raises(FileNotFoundError):
        VideoReader("non_existent_file.mp4")
