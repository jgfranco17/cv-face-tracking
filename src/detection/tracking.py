import logging
from abc import ABC, abstractmethod

import cv2

from src.detection.models import Face

logger = logging.getLogger(__name__)


class TrackerBase(ABC):
    """Abstract base class for trackers."""

    @abstractmethod
    def trace(self, frame: cv2.typing.MatLike) -> list[Face]:
        pass


class FaceTracker(TrackerBase):
    """A simple face tracker client using OpenCV's Haar cascades."""

    CASCADES_FILE: str = "haarcascade_frontalface_default.xml"
    BOX_COLOR: tuple[int, int, int] = (25, 255, 25)

    def __init__(self) -> None:
        cascade_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
        if not cascade_root:
            raise RuntimeError("OpenCV haarcascades path unavailable")

        cascade_path = f"{cascade_root}{self.CASCADES_FILE}"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def trace(self, frame: cv2.typing.MatLike) -> list[Face]:
        """Trace a face in frame.

        Args:
            frame (cv2.typing.MatLike): Image array representing the frame to be processed.

        Returns:
            list[Face]: List of detected faces.
        """
        image_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            image=image_grayscale, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)
        )

        detected_faces = []
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)
            tracked_face = Face(x, y, w, h)
            detected_faces.append(tracked_face)

        logger.debug(f"Traced {len(detected_faces)} faces in frame")
        return detected_faces
