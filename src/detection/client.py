import logging
from dataclasses import dataclass
from time import perf_counter

import cv2

from src.detection.camera import CaptureDevice
from src.detection.tracking import FaceTracker, TrackerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    duration: float
    frames_processed: int

    @property
    def frame_rate(self) -> float:
        return self.frames_processed / self.duration if self.duration > 0 else 0.0


class FaceDetectionClient:
    """Primary client entrypoint for face detection.

    Handles capturing frames from the specified capture device,
    processing them with the provided tracker, and displaying
    the live feed in a window.
    """

    def __init__(self, capture_device: CaptureDevice, tracker: TrackerBase) -> None:
        """Initialize a new client.

        Dependency injection allows us to control the client's
        interactions with real captures.

        Args:
            capture_device (CaptureDevice): Source of video frames.
            tracker (TrackerBase): Tracker used to process the frames.
        """
        self._capture_device = capture_device
        self._tracker = tracker
        self._is_running = False

        self._show_func = cv2.imshow

    @classmethod
    def new_default(cls) -> "FaceDetectionClient":
        capture_device = cv2.VideoCapture(0)
        tracker = FaceTracker()
        logger.info("Initialized default face-detection client")
        return cls(capture_device=capture_device, tracker=tracker)

    def run(self, window_name: str = "Live Feed") -> RunSummary:
        if not self._capture_device.isOpened():
            logger.error("Capture device is not opened")
            return RunSummary(duration=0.0, frames_processed=0)

        self._is_running = True
        start_time = perf_counter()
        frames_processed = 0
        while self._is_running:
            ret, frame = self._capture_device.read()
            if not ret:
                self._is_running = False
                logger.error("Failed to read frame from capture device")
                break

            self._tracker.trace(frame)
            self._show_func(window_name, frame)
            frames_processed += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._is_running = False
                logger.warning("User requested exit, shutting down")
                break

        self._capture_device.release()
        duration = perf_counter() - start_time
        return RunSummary(duration=duration, frames_processed=frames_processed)
