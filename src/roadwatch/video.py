"""Video input and output management."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from roadwatch.logger import get_logger
from roadwatch.types import VideoMetadata

logger = get_logger("roadwatch.video")


class VideoReader:
    """Safely opens and yields frames from video files with metadata."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.file_path}")

        self._cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self.file_path}")

        self.metadata = self._extract_metadata()
        logger.info(
            f"Opened video: {self.file_path.name} | Resolution: {self.metadata.width}x{self.metadata.height} | "
            f"FPS: {self.metadata.fps:.2f} | Frames: {self.metadata.total_frames} | Duration: {self.metadata.duration:.2f}s"
        )

    def _extract_metadata(self) -> VideoMetadata:
        assert self._cap is not None
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0.0:
            fps = 30.0  # Fallback default if video container lacks FPS header
        duration = total_frames / fps if total_frames > 0 else 0.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
        )

    def read_frames(
        self,
        frame_skip: int = 0,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yield (frame_index, timestamp_seconds, frame_bgr) tuples.
        Supports frame skipping and early stopping.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoCapture is closed.")

        frame_idx = 0
        yielded_count = 0

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            # Calculate timestamp based on frame index and FPS
            timestamp = frame_idx / self.metadata.fps

            if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
                frame_idx += 1
                continue

            yield frame_idx, timestamp, frame
            frame_idx += 1
            yielded_count += 1

            if max_frames is not None and yielded_count >= max_frames:
                break

    def close(self) -> None:
        """Release video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VideoWriter:
    """Encodes and writes processed frames to an output video file."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer: Optional[cv2.VideoWriter] = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self._writer.isOpened():
            # Try fallback to 'mp4v' if user-selected codec failed
            logger.warning(f"Codec '{codec}' failed. Attempting fallback to 'mp4v'.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

        if not self._writer.isOpened():
            raise IOError(f"Failed to initialize VideoWriter for path: {self.output_path}")

        logger.info(f"Initialized VideoWriter -> {self.output_path} ({width}x{height} @ {fps:.2f} FPS)")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame. Resizes frame if dimensions do not match."""
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError("VideoWriter is closed or uninitialized.")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._writer.write(frame)

    def close(self) -> None:
        """Safely flush and release the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"Finished writing video: {self.output_path}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
